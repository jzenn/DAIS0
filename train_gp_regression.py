import argparse
import os

import numpy as np
import torch
import torch.optim as optim

from src.distribution.vi import MeanFieldVariationalDistribution
from src.gp_regression.linalg import (
    analytic_fobs_given_data,
    get_analytic_joint,
    gp_ground_truth,
    rbf_1,
    rbf_2,
)
from src.gp_regression.model import GPAISModel, GPJoint
from src.sampling.betas import LearnableBetas
from src.sampling.dais import IdentityDAIS
from src.sampling.deltas import LearnableDeltas
from src.sampling.ld_momentum import LangevinMomentumDiffusionDAISZhang

sqrt_3 = torch.sqrt(torch.tensor(3.0))


def get_arguments():
    parser = argparse.ArgumentParser(description="GP Regression Experiment")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_particles", type=int, default=16)
    parser.add_argument("--n_transitions", type=int, default=16)
    parser.add_argument("--scaled_M", action="store_true")
    parser.add_argument("--mean_field", action="store_true")
    parser.add_argument("--importance_weighted", action="store_true")
    parser.add_argument("--do_not_apply_log_mean_exp_to_elbo", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=5_000)
    parser.add_argument("--observation_sigma", type=float, default=0.25)
    parser.add_argument("--n_observations", type=int, default=30)
    parser.add_argument("--n_x", type=int, help="number of points in domain")
    parser.add_argument("--kernel", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--path_to_save", type=str)
    args = parser.parse_args()
    return args


def unshuffle_mean_cov(mu_obs, mu_unobs, cov_obs, cov_unobs, index, inverse_index, n_x):
    mu_plot = torch.zeros(n_x)
    sigma_plot = torch.zeros(n_x)
    for i in range(n_x):
        if i in index:
            pos_index = torch.argmax((index == i).int()).item()
            mu_plot[i] = mu_obs[pos_index]
            sigma_plot[i] = torch.sqrt(cov_obs[pos_index, pos_index])
        else:
            assert i in inverse_index
            pos_inverse_index = torch.argmax((inverse_index == i).int()).item()
            mu_plot[i] = mu_unobs[pos_inverse_index]
            sigma_plot[i] = torch.sqrt(cov_unobs[pos_inverse_index, pos_inverse_index])

    return mu_plot, sigma_plot


def get_model(args, x, L_obs, K_obs):
    # variational distribution
    q = MeanFieldVariationalDistribution(
        dim=args.n_observations, diagonal_covariance=1.0, random_initialization=False
    )
    # target distribution
    p = GPJoint(
        prior_mean=torch.zeros(args.n_observations),
        prior_cholesky_factor=L_obs,
        prior_covariance=K_obs,
        observations=x,
        observation_scale=torch.tensor(args.observation_sigma),
    )
    # deltas
    deltas = LearnableDeltas(args)
    # betas
    betas = LearnableBetas(steps=args.n_transitions)
    # ais
    ais = LangevinMomentumDiffusionDAISZhang if not args.mean_field else IdentityDAIS
    # model
    model = GPAISModel(
        args=args,
        log_joint=p,
        log_variational=q,
        ais=ais(args=args, log_joint=p, log_variational=q, deltas=deltas, betas=betas),
    )
    return model


def main():
    # args
    args = get_arguments()

    # dimensionality of latent process
    args.zdim = args.n_observations

    # kernel
    if args.kernel == "rbf2":
        kernel = rbf_2
    elif args.kernel == "rbf1":
        kernel = rbf_1
    else:
        raise ValueError("Unknown kernel")

    # process, observations and ground truth
    (
        t,
        mu,
        K,
        L,
        x_t,
        x,
        index,
        inverse_index,
        K_obs,
        K_unobs,
        K_obs_unobs,
        K_unobs_obs,
        L_obs,
        mu_obs_plus,
        cov_obs_plus,
        c_alpha,
    ) = gp_ground_truth(kernel, args.n_x, args.n_observations, args.observation_sigma)

    model = get_model(args, x, L_obs, K_obs)

    # train
    torch.manual_seed(args.seed)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    max_iterations = args.max_iterations + 1

    # figure
    model_identifier = (
        "mf" + ("_iw" if args.importance_weighted else "")
        if args.mean_field
        else "dais"
    )

    path_to_save = os.path.join(
        args.path_to_save,
        f"{args.n_x}",
        f"{args.n_observations}",
        f"{args.kernel}",
        f"{args.n_particles}",
    )
    # create directory if not exists
    os.makedirs(path_to_save, exist_ok=True)
    os.makedirs(os.path.join(path_to_save, "models"), exist_ok=True)

    mu_obs_plus, cov_obs_plus = analytic_fobs_given_data(
        torch.zeros(args.n_observations), K_obs, args.observation_sigma, x
    )
    (
        mu_obs_plus_analytic,
        mu_unobs_plus_analytic,
        cov_obs_plus_analytic,
        cov_unobs_plus_analytic,
        cov_obs_unobs_plus_analytic,
        cov_unobs_obs_plus_analytic,
    ) = get_analytic_joint(
        mu_obs_plus,
        cov_obs_plus,
        torch.zeros(args.n_observations),
        torch.zeros(args.n_x - args.n_observations),
        K_obs,
        K_unobs,
        K_obs_unobs,
        K_unobs_obs,
    )
    mu_plot_analytic, sigma_plot_analytic = unshuffle_mean_cov(
        mu_obs_plus_analytic,
        mu_unobs_plus_analytic,
        cov_obs_plus_analytic,
        cov_unobs_plus_analytic,
        index,
        inverse_index,
        args.n_x,
    )

    # training
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        return_dict = model()
        loss = -return_dict["elbo"].mean()
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print(f"iteration: {iteration}, loss: {loss.item()}")

    save_results(
        K_obs,
        K_obs_unobs,
        K_unobs,
        K_unobs_obs,
        args,
        cov_obs_plus,
        index,
        inverse_index,
        model,
        model_identifier,
        mu_obs_plus,
        mu_obs_plus_analytic,
        mu_plot_analytic,
        mu_unobs_plus_analytic,
        path_to_save,
        sigma_plot_analytic,
        t,
        x,
        x_t,
    )


def save_results(
    K_obs,
    K_obs_unobs,
    K_unobs,
    K_unobs_obs,
    args,
    cov_obs_plus,
    index,
    inverse_index,
    model,
    model_identifier,
    mu_obs_plus,
    mu_obs_plus_analytic,
    mu_plot_analytic,
    mu_unobs_plus_analytic,
    path_to_save,
    sigma_plot_analytic,
    t,
    x,
    x_t,
):
    (
        mu_obs_plus_learned,
        mu_unobs_plus_learned,
        cov_obs_plus_learned,
        cov_unobs_plus_learned,
        cov_obs_unobs_plus_learned,
        cov_unobs_obs_plus_learned,
    ) = get_analytic_joint(
        # learned posterior mean
        model.q.mean.detach().cpu(),
        # learned posterior covariance
        torch.diag(model.q.log_diagonal.detach().exp().cpu()),
        torch.zeros(args.n_observations),
        torch.zeros(args.n_x - args.n_observations),
        K_obs,
        K_unobs,
        K_obs_unobs,
        K_unobs_obs,
    )
    mu_plot_learned, sigma_plot_learned = unshuffle_mean_cov(
        mu_obs_plus_learned,
        mu_unobs_plus_learned,
        cov_obs_plus_learned,
        cov_unobs_plus_learned,
        index,
        inverse_index,
        args.n_x,
    )
    for arr, arr_name in zip(
        [t, x_t, x, mu_plot_analytic, mu_plot_learned],
        ["t", "x_t", "x", "analytical_posterior_mean", "learned_posterior_mean"],
    ):
        np.save(
            os.path.join(
                path_to_save,
                "models",
                f"gp_{args.kernel}_{args.zdim}_{args.n_observations}_"
                f"{args.observation_sigma}_{args.n_particles}_"
                f"{args.n_transitions}_{model_identifier}_{arr_name}.npy",
            ),
            arr,
        )
    (
        _,
        _,
        cov_obs_plus_optimal_learned,
        cov_unobs_plus_optimal_learned,
        _,
        _,
    ) = get_analytic_joint(
        # analytical posterior mean
        mu_obs_plus,
        # optimal posterior covariance without any off-diagonal covariances
        torch.diag(torch.diag(cov_obs_plus)),
        torch.zeros(args.n_observations),
        torch.zeros(args.n_x - args.n_observations),
        K_obs,
        K_unobs,
        K_obs_unobs,
        K_unobs_obs,
    )
    _, sigma_plot_optimal_learned = unshuffle_mean_cov(
        mu_obs_plus_analytic,
        mu_unobs_plus_analytic,
        cov_obs_plus_optimal_learned,
        cov_unobs_plus_optimal_learned,
        index,
        inverse_index,
        args.n_x,
    )
    for arr, arr_name in zip(
        [sigma_plot_analytic, sigma_plot_learned, sigma_plot_optimal_learned],
        ["posterior_std", "learned_posterior_std", "diag_analytical_posterior_std"],
    ):
        np.save(
            os.path.join(
                path_to_save,
                "models",
                f"gp_{args.kernel}_{args.zdim}_{args.n_observations}_"
                f"{args.observation_sigma}_{args.n_particles}_"
                f"{args.n_transitions}_{model_identifier}_{arr_name}.npy",
            ),
            arr,
        )
    torch.save(
        model.state_dict(),
        os.path.join(
            path_to_save,
            "models",
            f"gp_{args.kernel}_{args.zdim}_{args.n_observations}_"
            f"{args.observation_sigma}_{args.n_particles}_"
            f"{args.n_transitions}_{model_identifier}.pt",
        ),
    )


if __name__ == "__main__":
    main()

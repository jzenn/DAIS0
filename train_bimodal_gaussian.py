import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim

from src.bimodal_gaussian.gaussian_mixture import (
    FixedBimodalGaussianMixture,
    FixedBimodalGaussianMixtureAISModel,
)
from src.distribution.vi import MeanFieldVariationalDistribution
from src.sampling.betas import LearnableBetas
from src.sampling.dais import IdentityDAIS
from src.sampling.deltas import LearnableDeltas
from src.sampling.ld_momentum import LangevinMomentumDiffusionDAISZhang

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_arguments():
    parser = argparse.ArgumentParser(description="Bimodal Gaussian Experiment")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_particles", type=int, default=32)
    parser.add_argument("--n_transitions", type=int, default=None)
    parser.add_argument("--zdim", type=int, default=None)
    parser.add_argument("--scaled_M", action="store_true")
    parser.add_argument("--mean_field", action="store_true")
    parser.add_argument("--importance_weighted", action="store_true")
    parser.add_argument("--do_not_apply_log_mean_exp_to_elbo", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=2500)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--path_to_save", type=str)
    args = parser.parse_args()
    return args


def get_model(args):
    # target distribution
    p = FixedBimodalGaussianMixture(
        means=[torch.zeros(args.zdim), torch.ones(args.zdim)],
        covariances=[torch.diag(torch.ones(args.zdim)) * args.sigma**2] * 2,
        weights=torch.tensor([0.5, 0.5]),
        device=device,
    ).to(device)

    # variational distribution
    q = MeanFieldVariationalDistribution(
        dim=args.zdim, diagonal_covariance=1.0, random_initialization=False
    ).to(device)
    with torch.no_grad():
        q.mean.data = q.mean.data + 0.5

    # deltas
    deltas = LearnableDeltas(args).to(device)

    # betas
    betas = LearnableBetas(steps=args.n_transitions).to(device)

    # ais
    if args.mean_field and not args.importance_weighted:
        ais = IdentityDAIS
        model_name = "mf"
    elif args.mean_field and args.importance_weighted:
        ais = IdentityDAIS
        model_name = "iw-mf"
    elif args.n_transitions is not None:
        ais = LangevinMomentumDiffusionDAISZhang
        model_name = "dais"
    else:
        raise RuntimeError("Unknown experiment.")

    # model
    model = FixedBimodalGaussianMixtureAISModel(
        args=args,
        log_joint=p,
        log_variational=q,
        ais=ais(
            args=args, log_joint=p, log_variational=q, deltas=deltas, betas=betas
        ).to(device),
        device=device,
    ).to(device)
    return model, model_name


def train_model(args, model):
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    losses = list()

    with torch.autograd.set_detect_anomaly(True):
        for iteration in range(args.max_iterations):
            optimizer.zero_grad()
            return_dict = model()
            loss = -return_dict["elbo"].mean()
            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                losses.append(loss.item())
            if iteration % 100 == 0:
                print(f"iteration: {iteration}, loss: {loss.item()}")
    return model


def main():
    # arguments
    args = get_arguments()

    # build model
    model, model_name = get_model(args)

    # train model
    model = train_model(args, model)

    # save results
    entropy = {
        model_name: {
            args.zdim: args.zdim / 2 * (1 + np.log(2.0 * np.pi))
            + 1 / 2 * model.q.log_diagonal.sum().item()
        }
    }
    distribution_params = {
        model_name: {
            args.zdim: {
                "mean": [float(m) for m in model.q.mean.detach().cpu().numpy()],
                "log_variance": [
                    float(m) for m in model.q.log_diagonal.detach().cpu().numpy()
                ],
            }
        }
    }
    path_to_save = os.path.join(
        args.path_to_save,
        f"{model_name}_entropies_{args.n_particles}_{args.n_transitions}_"
        f"{args.zdim}_{args.sigma}_{args.scaled_M}",
    )
    os.makedirs(args.path_to_save, exist_ok=True)
    with open(path_to_save + ".json", "w") as f:
        json.dump(entropy, f)
    with open(path_to_save.replace("entropies", "means_variances") + ".json", "w") as f:
        json.dump(distribution_params, f)


if __name__ == "__main__":
    main()

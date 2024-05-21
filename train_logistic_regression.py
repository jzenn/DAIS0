import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.heart_attack import HeartAttackDataset
from src.data.heart_disease import HeartDiseaseDataset
from src.data.ionosphere import IonosphereDataset
from src.data.loan import LoanDataset
from src.data.sonar import SonarDataset
from src.distribution.vi import MeanFieldVariationalDistribution
from src.logistic_regression.model import (
    BayesianLogisticRegressionLogJoint,
    LogisticRegressionAISModel,
)
from src.logistic_regression.sampling import HMCSampler
from src.sampling.betas import LearnableBetas
from src.sampling.dais import IdentityDAIS
from src.sampling.deltas import LearnableDeltas
from src.sampling.ld_momentum import LangevinMomentumDiffusionDAISZhang


def get_arguments():
    parser = argparse.ArgumentParser(description="Logistic Regression Experiment")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_particles", type=int, default=16)
    parser.add_argument("--n_transitions", type=int, default=16)
    parser.add_argument("--scaled_M", action="store_true")
    parser.add_argument("--mean_field", action="store_true")
    parser.add_argument("--importance_weighted", action="store_true")
    parser.add_argument("--do_not_apply_log_mean_exp_to_elbo", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--path_to_save", type=str)
    parser.add_argument("--hmc", action="store_true", help="Generate HMC samples")
    parser.add_argument("--hmc_step_size", type=float, default=0.005)
    parser.add_argument("--hmc_num_leapfrog_steps", type=int, default=100)
    parser.add_argument("--hmc_num_samples", type=int, default=500)
    parser.add_argument("--hmc_take_every_n_sample", type=int, default=10)
    parser.add_argument("--hmc_burnin", type=int, default=500)
    args = parser.parse_args()
    return args


def get_accuracy(zs, x, y):
    probs = torch.sigmoid(
        -(zs[..., :-1] * x.unsqueeze(1) + zs[..., -1].unsqueeze(-1)).sum(-1)
    )
    return ((probs > 0.5).int() == y.unsqueeze(-1)).int().float().mean()


def sample(args):
    train_data_loader, test_data_loader = get_data_loaders(args)
    p = BayesianLogisticRegressionLogJoint(
        torch.zeros(args.zdim - 1), torch.eye(args.zdim - 1)
    )

    # sample
    hmc_sampler = HMCSampler(args, p)
    for x, y in train_data_loader:
        samples, hamiltonians, alphas, accepts = hmc_sampler.sample((args.zdim,), x, y)

    # test accuracies
    test_accuracies = list()
    with torch.no_grad():
        for sample in samples:
            for x, y in test_data_loader:
                test_accuracy = get_accuracy(sample, x, y)
                test_accuracies.append(test_accuracy.item())

    # save
    path_to_save = os.path.join(
        args.path_to_save, "hmc", f"{args.dataset}"
    )
    os.makedirs(path_to_save, exist_ok=True)
    filename_prefix = (
        "hmc_samples_"
        f"{args.hmc_step_size}_"
        f"{args.hmc_num_leapfrog_steps}_"
        f"{args.hmc_num_samples}_"
        f"{args.hmc_take_every_n_sample}_"
        f"{args.hmc_burnin}"
    )

    # figure
    fig, ax = plt.subplots(3, 1, figsize=(1 * 5, 3 * 5))
    ax[0].plot(hamiltonians)
    ax[0].set_title("Hamiltonians")
    ax[1].plot(np.minimum(alphas, 1.0))
    ax[1].set_title("Alphas")
    ax[2].plot(test_accuracies)
    ax[2].set_title("Test accuracies")
    fig.savefig(os.path.join(path_to_save, filename_prefix + ".png"))
    plt.close(fig)

    # samples
    torch.save(
        samples,
        os.path.join(path_to_save, filename_prefix + "_samples.pt"),
    )
    torch.save(
        alphas,
        os.path.join(path_to_save, filename_prefix + "_alphas.pt"),
    )
    torch.save(
        test_accuracies,
        os.path.join(path_to_save, filename_prefix + "_test_accuracies.pt"),
    )
    torch.save(
        hamiltonians,
        os.path.join(path_to_save, filename_prefix + "_hamiltonians.pt"),
    )


def get_data_loaders(args):
    # dataset
    if args.dataset == "ionosphere":
        ionosphere_dataset = IonosphereDataset(args.data_path)
        dataset_dim = 34
        args.zdim = dataset_dim + 1
        training_set_size = int(0.8 * len(ionosphere_dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(
            ionosphere_dataset,
            [training_set_size, len(ionosphere_dataset) - training_set_size],
            generator=torch.Generator().manual_seed(0),
        )
    elif args.dataset == "sonar":
        sonar_dataset = SonarDataset(args.data_path)
        dataset_dim = 60
        args.zdim = dataset_dim + 1
        training_set_size = int(0.8 * len(sonar_dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(
            sonar_dataset,
            [training_set_size, len(sonar_dataset) - training_set_size],
            generator=torch.Generator().manual_seed(0),
        )
    elif args.dataset == "heart-disease":
        heart_disease_dataset = HeartDiseaseDataset(args.data_path)
        dataset_dim = 15
        args.zdim = dataset_dim + 1
        training_set_size = int(0.8 * len(heart_disease_dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(
            heart_disease_dataset,
            [training_set_size, len(heart_disease_dataset) - training_set_size],
            generator=torch.Generator().manual_seed(0),
        )
    elif args.dataset == "heart-attack":
        heart_attack_dataset = HeartAttackDataset(args.data_path)
        dataset_dim = 13
        args.zdim = dataset_dim + 1
        training_set_size = int(0.8 * len(heart_attack_dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(
            heart_attack_dataset,
            [training_set_size, len(heart_attack_dataset) - training_set_size],
            generator=torch.Generator().manual_seed(0),
        )
    elif args.dataset == "loan":
        loan_dataset = LoanDataset(args.data_path)
        dataset_dim = 11
        args.zdim = dataset_dim + 1
        training_set_size = int(0.8 * len(loan_dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(
            loan_dataset,
            [training_set_size, len(loan_dataset) - training_set_size],
            generator=torch.Generator().manual_seed(0),
        )
    else:
        raise RuntimeError("Unknown dataset.")

    # data loader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_data_loader, test_data_loader


def get_ais_model(args):
    # target distribution
    p = BayesianLogisticRegressionLogJoint(
        torch.zeros(args.zdim - 1), torch.eye(args.zdim - 1)
    )

    # variational distribution
    q = MeanFieldVariationalDistribution(
        args.zdim, diagonal_covariance=1.0, random_initialization=False
    )

    # deltas
    deltas = LearnableDeltas(args)

    # betas
    betas = LearnableBetas(steps=args.n_transitions)

    # ais
    ais = LangevinMomentumDiffusionDAISZhang if not args.mean_field else IdentityDAIS

    # model
    model = LogisticRegressionAISModel(
        args=args,
        log_joint=p,
        log_variational=q,
        ais=ais(args=args, log_joint=p, log_variational=q, deltas=deltas, betas=betas),
    )
    return model


def train_model(args, model, optimizer, train_data_loader, test_data_loader):
    torch.manual_seed(args.seed)

    losses = list()
    initial_train_accuracies = list()
    train_accuracies = list()
    test_losses = list()
    test_accuracies = list()
    initial_test_accuracies = list()
    with torch.autograd.set_detect_anomaly(False):
        for iteration in range(args.max_iterations):
            # training
            for x, y in train_data_loader:
                optimizer.zero_grad()
                return_dict = model(x, y)
                with torch.no_grad():
                    train_accuracy = get_accuracy(return_dict["last_z"], x, y)
                    initial_train_accuracy = get_accuracy(model.q.mean, x, y)
                train_loss = -return_dict["elbo"].mean()
                train_loss.backward()
                optimizer.step()

            # testing
            for x, y in test_data_loader:
                return_dict = model(x, y)
                with torch.no_grad():
                    test_accuracy = get_accuracy(return_dict["last_z"], x, y)
                    initial_test_accurcy = get_accuracy(model.q.mean, x, y)
                test_loss = -return_dict["elbo"].mean()

            # logging
            if iteration % 250 == 0:
                losses.append(train_loss.item())
                train_accuracies.append(train_accuracy)
                initial_train_accuracies.append(initial_train_accuracy)
                test_losses.append(test_loss.item())
                test_accuracies.append(test_accuracy)
                initial_test_accuracies.append(initial_test_accurcy)
                print(
                    f"iteration: {iteration:.4f}\t"
                    f"train loss: {train_loss.item():.4f}\t"
                    f"test loss: {test_loss.item():.4f}\t"
                    f"train accuracy: {train_accuracy.item():.4f}\t"
                    f"test accuracy: {test_accuracy.item():.4f}\t"
                    f"initial train accuracy: {initial_train_accuracy.item():.4f}\t"
                    f"initial test accuracy: {initial_test_accurcy.item():.4f}\t"
                )
    return (
        initial_test_accuracies,
        initial_train_accuracies,
        losses,
        test_accuracies,
        test_losses,
        train_accuracies,
    )


def main():
    # arguments
    args = get_arguments()

    if args.hmc:
        sample(args)
        exit(0)

    # data
    train_data_loader, test_data_loader = get_data_loaders(args)

    # model
    model = get_ais_model(args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    (
        initial_test_accuracies,
        initial_train_accuracies,
        losses,
        test_accuracies,
        test_losses,
        train_accuracies,
    ) = train_model(args, model, optimizer, train_data_loader, test_data_loader)

    # save results
    model_identifier = f"{args.dataset}_" + (
        ("mf" + ("_iw" if args.importance_weighted else ""))
        if args.mean_field
        else "dais"
    )

    path_to_save = os.path.join(
        args.path_to_save,
        f"{args.dataset}",
        f"{args.n_particles}",
        f"{args.n_transitions}",
    )

    # create directory if not exists
    os.makedirs(path_to_save, exist_ok=True)
    os.makedirs(os.path.join(path_to_save, "models"), exist_ok=True)
    os.makedirs(os.path.join(path_to_save, "plots"), exist_ok=True)

    # model parameters
    torch.save(
        model.state_dict(),
        os.path.join(
            path_to_save,
            "models",
            f"log_reg_"
            f"{args.n_particles}_"
            f"{args.n_transitions}_"
            f"{model_identifier}.pt",
        ),
    )

    # figure summarizing run
    fig, ax = plt.subplots(3, 1, figsize=(1 * 5, 2 * 5))

    # losses
    ax[0].plot(losses, label="train", color="C0")
    ax[0].plot(test_losses, label="test", color="C1")
    ax[0].legend()
    ax[0].set_title("Losses")

    # train accuracies
    ax[1].plot(train_accuracies, label="final", color="C2")
    ax[1].plot(initial_train_accuracies, label="initial", color="C3")
    ax[1].legend()
    ax[1].set_title("Train Accuracies")

    # test accuracies
    ax[2].plot(test_accuracies, label="final", color="C2")
    ax[2].plot(initial_test_accuracies, label="initial", color="C3")
    ax[2].legend()
    ax[2].set_title("Test Accuracies")

    fig.savefig(
        os.path.join(
            path_to_save,
            "plots",
            f"log_reg_"
            f"{args.n_particles}_"
            f"{args.n_transitions}_"
            f"{model_identifier}.pdf",
        ),
    )
    plt.close(fig)


if __name__ == "__main__":
    main()

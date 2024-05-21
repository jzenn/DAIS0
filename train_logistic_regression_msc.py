# all of this code is taken and adapted from
# https://github.com/blei-lab/markovian-score-climbing
#
import argparse
import os

import autograd.numpy as np
import autograd.numpy.random as npr
import torch
from autograd import grad
from autograd.misc.optimizers import adam

from src.data.ionosphere import IonosphereDataset
from src.data.sonar import SonarDataset
from src.msc.model import generate_samples, log_posteriorapprox


def get_arguments():
    parser = argparse.ArgumentParser(description="Logistic Regression Experiment")
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n_particles", type=int, default=16)
    parser.add_argument("--ionosphere", action="store_true")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--path_to_save", type=str)
    args = parser.parse_args()
    return args


def get_dataset(args):
    if args.ionosphere:
        dataset = IonosphereDataset(args.data_path)
        dataset_dim = 34
    else:
        dataset = SonarDataset(args.data_path)
        dataset_dim = 60
    return dataset, dataset_dim


def load_data(args):
    dataset, dataset_dim = get_dataset(args)
    training_set_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [training_set_size, len(dataset) - training_set_size],
        generator=torch.Generator().manual_seed(0),
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=True
    )
    for x, y in train_data_loader:
        train_data = (
            y.numpy().reshape(-1, 1),
            np.insert(x.reshape(-1, dataset_dim).numpy(), 0, 1, axis=1),
        )
    for x, y in test_data_loader:
        test_data = (
            y.numpy().reshape(-1, 1),
            np.insert(x.reshape(-1, dataset_dim).numpy(), 0, 1, axis=1),
        )
    return train_data, test_data, len(dataset), dataset_dim + 1


def get_objective(seed, train_data, nz, S):
    def objective(params, iter, verbose=False):
        global zprev
        w, z, zprev = generate_samples(params, seed, zprev, verbose, train_data, nz, S)
        return -np.mean(log_posteriorapprox(params, z) * w)

    return objective


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


def main():
    # arguments
    args = get_arguments()

    seed = npr.RandomState(args.seed)
    npr.seed(args.seed)
    init_scale = 0.1

    train_data, test_data, ny, nz = load_data(args)

    objective = get_objective(seed, train_data, nz, args.n_particles)
    objective_grad = grad(get_objective)

    init_mu = init_scale * seed.normal(size=nz)
    init_logsigma = 0.5 + init_scale * seed.normal(size=nz)

    def print_perf(params, iter, grad):
        global zprev
        m, ls = params
        mu_vec[iter] = m
        logsigma_vec[iter] = ls
        if iter % 100 == 0:
            bound = np.mean(objective(params, iter, False))
            message = "{:15}|{:20}|".format(iter, bound)
            print(message)

    init_params = (init_mu, init_logsigma)
    zprev = init_scale * seed.normal(size=nz)

    mu_vec = np.zeros((args.max_iterations, nz))
    logsigma_vec = np.zeros((args.max_iterations, nz))

    print("     Epoch     |    Objective  ")
    optimized_params = adam(
        objective_grad,
        init_params,
        step_size=args.lr,
        num_iters=args.max_iterations,
        callback=print_perf,
    )

    # Prediction error
    mu_opt = np.mean(mu_vec[-150:], axis=0)
    sigma_opt = np.diag(np.mean(np.exp(2.0 * logsigma_vec[-150:]), axis=0))

    path_to_save = os.path.join(
        args.path_to_save,
        "ionosphere" if args.ionosphere else "sonar",
        f"{args.n_particles}",
    )

    # create directory if not exists
    os.makedirs(path_to_save, exist_ok=True)
    os.makedirs(os.path.join(path_to_save, "model"), exist_ok=True)
    os.makedirs(os.path.join(path_to_save, "results"), exist_ok=True)

    filename_prefix = (
        "msc_bayesian_logistic_regression_"
        f"{args.dataset}_"
        f"{args.n_particles}_"
        f"{args.seed}_"
    )

    np.save(
        os.path.join(
            path_to_save,
            "model",
            f"{filename_prefix}_mu_opt.npy",
        ),
        mu_opt,
    )
    np.save(
        os.path.join(
            path_to_save,
            "model",
            f"{filename_prefix}_sigma_opt.npy",
        ),
        sigma_opt,
    )


if __name__ == "__main__":
    main()

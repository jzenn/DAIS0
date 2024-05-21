# all of this code is taken and adapted from
# https://github.com/blei-lab/markovian-score-climbing
#
import autograd.numpy as np
from autograd.extend import notrace_primitive


def discretesampling(w, seed):
    u = seed.rand()
    bins = np.cumsum(w)
    return np.digitize(u, bins)


def log_likelihood(x, y, z):
    logits = x @ z
    log_likelihood = y * logits - np.logaddexp(0, logits)
    return log_likelihood.sum(0)


def log_prior(latent):
    # latent is array (nz, S)
    return np.sum(-0.5 * latent**2 - 0.5 * np.log(2.0 * np.pi), axis=0)


def log_posteriorapprox(params, latent):
    mu, log_sigma = params
    return np.sum(
        -0.5 * (latent - mu[:, None]) ** 2 / np.exp(2.0 * log_sigma[:, None])
        - 0.5 * np.log(2.0 * np.pi)
        - log_sigma[:, None],
        axis=0,
    )


def log_weights(params, latent, data):
    y_data, x_data = data
    return (
        log_prior(latent)
        + log_likelihood(x_data, y_data, latent)
        - log_posteriorapprox(params, latent)
    )


@notrace_primitive
def generate_samples(params, seed, zprev, verbose, train_data, nz, S):
    mu, log_sigma = params

    # Propose values
    z = mu[:, None] + np.exp(log_sigma[:, None]) * seed.normal(size=(nz, S))
    z[:, 0] = zprev

    # Compute weights
    logw = log_weights(params, z, train_data)
    maxLogW = np.max(logw)
    uw = np.exp(logw - maxLogW)
    w = uw / np.sum(uw)

    # Sample new conditionals
    idx = discretesampling(w, seed)

    if verbose:
        print("ESS: " + str(1.0 / np.sum(w**2)))
    return w, z, z[:, idx]

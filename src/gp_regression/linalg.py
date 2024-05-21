import torch


def rbf(t, s, rho):
    return torch.exp(-((t - s) ** 2) / (2 * rho**2))


def rbf_2(t, s):
    return rbf(t, s, 0.8)


def rbf_1(t, s):
    return rbf(t, s, 3.0)


def analytic_fobs_given_data(prior_mean, prior_cov, data_scale, data):
    mu_obs_plus = prior_mean + prior_cov @ torch.linalg.solve(
        prior_cov + data_scale**2 * torch.eye(prior_cov.shape[0]), (data - prior_mean)
    )
    cov_obs_plus = prior_cov - prior_cov @ torch.linalg.solve(
        prior_cov + data_scale**2 * torch.eye(prior_cov.shape[0]), prior_cov
    )
    return mu_obs_plus, cov_obs_plus


def analytic_funobs_given_fobs(
    f_obs,
    prior_mean_obs,
    prior_mean_unobs,
    cov_obs,
    cov_unobs,
    cov_obs_unobs,
    cov_unobs_obs,
):
    mu_plus = prior_mean_unobs + cov_unobs_obs @ torch.linalg.solve(
        cov_obs, f_obs - prior_mean_obs
    )
    cov_plus = cov_unobs - cov_unobs_obs @ torch.linalg.solve(cov_obs, cov_obs_unobs)
    return mu_plus, cov_plus


def get_analytic_joint(
    mu_plus,
    sigma_plus,
    prior_mean_obs,
    prior_mean_unobs,
    cov_obs,
    cov_unobs,
    cov_obs_unobs,
    cov_unobs_obs,
):
    mu_obs_plus = mu_plus
    mu_unobs_plus = (
        cov_unobs_obs @ torch.linalg.solve(cov_obs, mu_plus)
        + cov_unobs_obs @ torch.linalg.solve(cov_obs, prior_mean_obs)
        + prior_mean_unobs
    )
    cov_obs_plus = sigma_plus
    cov_obs_unobs_plus = sigma_plus @ torch.linalg.solve(cov_obs.T, cov_unobs_obs.T)
    cov_unobs_obs_plus = cov_unobs_obs @ torch.linalg.solve(cov_obs, sigma_plus)
    cov_unobs_plus = (
        cov_unobs_obs
        @ torch.linalg.solve(cov_obs, sigma_plus)
        @ torch.linalg.solve(cov_obs.T, cov_unobs_obs.T)
        + cov_unobs
        - cov_unobs_obs @ torch.linalg.solve(cov_obs, cov_obs_unobs)
    )
    return (
        mu_obs_plus,
        mu_unobs_plus,
        cov_obs_plus,
        cov_unobs_plus,
        cov_obs_unobs_plus,
        cov_unobs_obs_plus,
    )


def separate_cov_by_index(K, index):
    inverse_index = torch.tensor([i for i in range(K.shape[0]) if i not in index])
    K_obs = K[index.unsqueeze(0), index.unsqueeze(1)]
    K_unobs = K[inverse_index.unsqueeze(0), inverse_index.unsqueeze(1)]
    K_unobs_obs = K[index.unsqueeze(0), inverse_index.unsqueeze(1)]
    K_obs_unobs = K[inverse_index.unsqueeze(0), index.unsqueeze(1)]
    return K_obs, K_unobs, K_obs_unobs, K_unobs_obs


def gp_ground_truth(kernel, N, n, observation_sigma):
    # set seed
    torch.manual_seed(0)

    # domain
    t = torch.linspace(0, 10.0, N)
    xx = torch.repeat_interleave(t, N).reshape(N, N)
    yy = torch.tile(t, (N,)).reshape(N, N)

    # kernel matrix
    K = kernel(xx, yy) + torch.eye(xx.shape[0]) * 1e-5
    L = torch.linalg.cholesky(K)

    # observations
    torch.manual_seed(0)
    xs = torch.randn(N)
    Lxs = L @ xs
    index = torch.randperm(N)[:n]
    inverse_index = torch.tensor([i for i in range(N) if i not in index])

    x = torch.zeros(n)
    x_t = torch.zeros(n)
    for i, j in enumerate(index):
        x[i] = Lxs[j] + observation_sigma * torch.randn(1)
        x_t[i] = t[j]

    K_obs, K_unobs, K_obs_unobs, K_unobs_obs = separate_cov_by_index(K, index)
    mu_obs_plus, cov_obs_plus = analytic_fobs_given_data(
        torch.zeros(n), K_obs, observation_sigma, x
    )

    L_obs = torch.linalg.cholesky(K_obs)

    alpha = 0.975
    c_alpha = torch.distributions.Normal(0, 1).icdf(torch.tensor(alpha)).item()
    return (
        t,
        Lxs,
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
    )

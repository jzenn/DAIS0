import torch


class HMCSampler:
    def __init__(self, args, log_joint):
        self.log_joint = log_joint
        self.step_size = args.hmc_step_size
        self.num_steps = args.hmc_num_leapfrog_steps
        self.num_samples = args.hmc_num_samples
        self.take_every_n_sample = args.hmc_take_every_n_sample
        self.burn_in = args.hmc_burnin

    def gradient_log_joint(self, z, x, y):
        grad = torch.autograd.grad(
            self.log_joint.log_prob(z, x, y), z, create_graph=True
        )[0]
        return grad

    def leapfrog(self, z, r, x, y):
        grad = self.gradient_log_joint(z, x, y)
        r = r + self.step_size / 2 * grad

        for j in range(self.num_steps):
            z = z + self.step_size * r
            grad = self.gradient_log_joint(z, x, y)
            r = r + self.step_size * grad

        r = r - self.step_size / 2 * grad

        return z, r

    def sample(self, param_shape, x, y):
        samples = list()
        hamiltonians = list()
        alphas = list()
        accepts = 0

        z = torch.randn(param_shape, requires_grad=True)

        for i in range(self.take_every_n_sample * self.num_samples + self.burn_in):
            r = torch.randn(param_shape)

            hamiltonian = (r**2).sum() / 2 - self.log_joint.log_prob(z, x, y)

            z_new, r_new = self.leapfrog(z.clone(), r.clone(), x, y)

            new_hamiltonian = (r_new**2).sum() / 2 - self.log_joint.log_prob(
                z_new, x, y
            )

            alpha = torch.exp(-(new_hamiltonian - hamiltonian))
            alphas.append(alpha.item())
            if torch.rand(1) < alpha:
                accepts += 1
                z = z_new.clone().detach().requires_grad_(True)

            if i >= self.burn_in and i % self.take_every_n_sample == 0:
                samples.append(z.detach().cpu())

            hamiltonians.append(hamiltonian.item())

        return torch.stack(samples), hamiltonians, alphas, accepts

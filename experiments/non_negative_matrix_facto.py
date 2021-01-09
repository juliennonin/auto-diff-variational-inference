#%%
%load_ext autoreload
%autoreload 2

#%%
from autograd import grad, elementwise_grad, jacobian
import autograd.numpy as np
from autograd.scipy import stats as sps
import matplotlib.pyplot as plt

from scipy.io import loadmat
from advi import ADVI

# %%
class GammaPoissonModel:

    def __init__(self, data, U, I, K=10, a=1, b=1, c=1, d=1):
        self.data = data

        self.K = K
        self.U = U
        self.I = I

        self.dim = self.K * (self.U + self.I)

        self.theta_shape, self.theta_scale = a, b
        self.beta_shape, self.beta_scale = c, d

    def get_params(self, params):
        assert len(params) == self.dim
        thetas, betas = np.split(params, [self.U  * self.K])
        return thetas, betas

    def _log_priors(self, params):
        thetas, betas = self.get_params(params)
        thetas_log_prior = np.sum(sps.gamma.logpdf(thetas / self.theta_scale, self.theta_shape) - np.log(self.theta_scale))
        betas_log_prior = np.sum(sps.gamma.logpdf(betas / self.beta_scale, self.beta_shape) - np.log(self.beta_scale))
        log_prior = thetas_log_prior + betas_log_prior
        return log_prior

    def _log_likelihood(self, params):
        thetas, betas = self.get_params(params)
        thetas = thetas.reshape(self.U, self.K)
        betas = betas.reshape(self.K, self.I)

        y_lam = np.matmul(thetas, betas) # .flatten()
        # print("Dom", thetas.shape, betas.shape)
        assert y_lam.shape == (self.U, self.I)
        return np.sum(sps.poisson.logpmf(self.data, y_lam))

    def log_joint(self, params):
        return self._log_priors(params) + self._log_likelihood(params)

# %%
%%time
if __name__ == "__main__":
    from scipy.special import expit
    n_train = 100
    data = loadmat("data/frey_rawface.mat")["ff"].T[:n_train]
    Nx, Ny = 28, 20

    model = GammaPoissonModel(data, U=n_train, I=Nx*Ny)

    def inv_T(zeta):
        return np.logaddexp(zeta, 0)  # T¹(zeta) -> log(exp(zeta) + 1)

    advi = ADVI(model, inv_T)
    advi.log_jac_inv_T = lambda zeta: np.sum(-np.logaddexp(-zeta, 0))
    advi.grad_log_jac_inv_T = lambda zeta: expit(-zeta)

    # Load already trained advi or train it!
    advi.mu = np.loadtxt("data/nmf_advi_mu.txt")
    advi.omega = np.loadtxt("data/nmf_advi_omega.txt")
    # advi.run(learning_rate=0.5)

    # np.savetxt("data/nmf_advi_mu.txt", advi.mu)
    # np.savetxt("data/nmf_advi_omega.txt", advi.omega)

    zeta = np.random.normal(advi.mu, np.exp(advi.omega))
    params = inv_T(zeta)
    thetas, betas = model.get_params(zeta)
    thetas = thetas.reshape(model.U, model.K)
    betas = betas.reshape(model.K, model.I)

    fig, axs = plt.subplots(2, 5, figsize=(12, 5))
    axs = axs.flatten()
    for i in range(10):
        axs[i].imshow(betas[i].reshape(28, 20), cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(fr"$\beta_{{{i}}}$")

    u = 42
    plt.figure()
    plt.subplot(121)
    plt.imshow((thetas[u] @ betas).reshape(28, 20), cmap="gray")
    plt.axis("off")
    plt.title(fr"$\theta_{{{u}}}\cdot\beta$")
    plt.subplot(122)
    plt.imshow(data[u].reshape(28, 20), cmap="gray")
    plt.title(fr"$Y_{{{u}, true}}$")
    plt.axis("off")
    print(f"theta_{u}: {thetas[u]}")



# %%

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

    def __init__(self, data, K=10, U=28, I=20, a=1, b=1, c=1, d=1):
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

        y_lam = np.matmul(thetas, betas).flatten()
        assert len(y_lam) == self.U * self.I
        return np.sum(sps.poisson.logpmf(self.data, y_lam))

    def log_joint(self, params):
        return self._log_priors(params) + self._log_likelihood(params)

# %%
if __name__ == "__main__":
    data = loadmat("data/frey_rawface.mat")["ff"].T[10:]

    model = GammaPoissonModel(data)

    def inv_T(zeta):
        return np.logaddexp(zeta, 0)  # T¹(zeta) -> log(exp(zeta) + 1)

    advi = ADVI(model, inv_T)
    advi.run(learning_rate=0.5)


# %%

#%%
from autograd import grad, elementwise_grad, jacobian
import autograd.numpy as np
from autograd.scipy import stats as sps

import matplotlib.pyplot as plt
from IPython.display import display, Math, Latex


# %%
class Model:
    def __init__(self):
        pass


class LinearModel(Model):
    """
        y ~ Normal(X⋅beta, sigma)
        beta_k ~ Normal(beta_mu, beta_std)
        sigma ~ Gamma(sigma_shape, sigma_scale)
    """
    def __init__(self, X, y, prior_beta_mu, prior_beta_std, prior_sigma_shape, prior_sigma_scale):
        self.x = X
        self.y = y

        self.dim = self.x.shape[1] + 2  # (intercept (beta0), betas, sigma)  
        self.N = self.x.shape[0]
        
        # Prior hyperparameters
        # self.betas_mu = prior_beta_mu * np.ones(self.dim - 1)
        # self.betas_std = prior_beta_std * np.ones(self.dim - 1)
        self.betas_mu = np.full(self.dim - 1, prior_beta_mu)
        self.betas_std = np.full(self.dim - 1, prior_beta_std)
        self.sigma_shape = prior_sigma_shape
        self.sigma_scale = prior_sigma_scale

    def params(self, theta):
        # [TODO] Change it into a wrapper to convert theta into (betas, sigma) 
        assert len(theta) == 3
        return theta[:2], theta[2]
    
    def _log_priors(self, theta):
        betas, sigma = self.params(theta)
        betas_log_prior = sps.norm.logpdf(betas, self.betas_mu, self.betas_std).sum()
        sigma_log_prior = sps.gamma.logpdf(sigma / self.sigma_scale, self.sigma_shape) - np.log(self.sigma_scale)
        return betas_log_prior + sigma_log_prior

    def _log_likelihood(self, theta):
        betas, sigma = self.params(theta)
        x_ones = np.hstack([np.ones((self.N, 1)), self.x])
        y_hat = x_ones @ betas
        return sps.norm.logpdf(self.y, y_hat, sigma).sum()

    def log_joint(self, theta):
        return self._log_priors(theta) + self._log_likelihood(theta)

    def describe(self):
        display(Math(fr"""
            y_i \sim \mathcal{{N}}(\mathbf{{\beta}} \mathbf{{X}}_i, \sigma) \\
            \beta_k \sim \mathcal{{N}}(m, s^2) \\
            \sigma \sim \mathcal{{G}}(\alpha, \lambda)
            """))

# %%
class ADVI:
    def __init__(self, model, inv_T):
        self.model = model
        self.inv_T = inv_T

        ## Compute Gradients
        self.grad_log_joint = elementwise_grad(model.log_joint)
        self.grad_inv_T = elementwise_grad(inv_T)
        
        jacobian_det_inv_T = lambda zeta: np.linalg.det(jacobian(inv_T)(zeta))
        self.grad_log_jac_inv_T = elementwise_grad(lambda zeta: np.log(np.abs(jacobian_det_inv_T(zeta))))

        # To optimize
        self.mu = np.ones(self.model.dim)
        self.omega = np.ones(self.model.dim)

    def _nabla_mu_inside_expect(self, eta):
        assert len(eta) == self.model.dim
        zeta = eta * np.exp(self.omega) + self.mu
        theta = self.inv_T(zeta)

        grad_log_joint_eval = self.grad_log_joint(theta)
        grad_inv_T_eval = self.grad_inv_T(zeta)
        grad_log_jac_inv_T_eval = self.grad_log_jac_inv_T(zeta) 
        return grad_log_joint_eval * grad_inv_T_eval + grad_log_jac_inv_T_eval

    def _nabla_omega_inside_expect(self, nabla_mu_eval, eta):
        return np.dot(nabla_mu_eval, eta) * np.exp(self.omega) + 1

    def run(self):
        # Stochastic optimization
        pass

# %%

N, d = 50, 1
betas_true = np.array([5, -3])
sigma_true = 2.5

X = np.random.normal(0, 2, (N, d))
X_ones = np.hstack([np.ones((N, 1)), X])
y = X_ones @ betas_true + np.random.normal(0, sigma_true, N)

model = LinearModel(X, y, 0, 10, 1, 2)
model.describe()

def inv_T(zeta):
    return np.array([*zeta[:-1], np.exp(zeta[-1])], dtype=float)

advi = ADVI(model, inv_T)
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
        self.dim = self.model.dim

        ## Compute Gradients
        self.grad_log_joint = elementwise_grad(model.log_joint)  # θ -> ∇log p(x, θ)
        self.grad_inv_T = elementwise_grad(inv_T)  # ζ -> ∇T⁻¹(ζ)
        
        jacobian_det_inv_T = lambda zeta: np.linalg.det(jacobian(inv_T)(zeta))
        self.log_jac_inv_T = lambda zeta: np.log(np.abs(jacobian_det_inv_T(zeta)))  # ζ -> log|det J_{T⁻¹}(ζ)| 
        self.grad_log_jac_inv_T = elementwise_grad(self.log_jac_inv_T)  # ζ -> ∇log|det J_{T⁻¹}(ζ)| 

        # To optimize
        self.mu = np.zeros(self.dim)
        self.omega = np.zeros(self.dim)
        self.history = {"mu": [], "omega": [], "elbo": []}

    def update_params(self, mu, omega):
        # [TODO] Create setter for properties mu and omega
        assert mu.shape[0] == self.dim
        assert omega.shape[0] == self.dim
        self.mu, self.omega = mu, omega
        self.history["mu"].append(mu)
        self.history["omega"].append(omega)

    def _zeta(self, eta):
        return eta * np.exp(self.omega) + self.mu

    def _theta(self, zeta):
        return self.inv_T(zeta)

    def _nabla_mu_inside_expect(self, eta):
        assert len(eta) == self.dim
        zeta = self._zeta(eta)
        theta = self._theta(zeta)

        grad_log_joint_eval = self.grad_log_joint(theta)
        grad_inv_T_eval = self.grad_inv_T(zeta)
        grad_log_jac_inv_T_eval = self.grad_log_jac_inv_T(zeta) 
        return grad_log_joint_eval * grad_inv_T_eval + grad_log_jac_inv_T_eval

    def _nabla_omega_inside_expect(self, nabla_mu_eval, eta):
        # [TODO] Is it an elementwise product between coeffs of the gradients?
        return nabla_mu_eval * eta * np.exp(self.omega) + 1

    def _gradients_approximate(self, M):
        # Draw M samples η from the standard multivariate Gaussian N(0,I)
        nabla_mu = np.zeros(self.dim)
        nabla_omega = np.zeros(self.dim)
        for m in range(M):
            eta = np.random.normal(size=self.dim)
            nabla_mu_eval = self._nabla_mu_inside_expect(eta)
            nabla_mu += nabla_mu_eval
            nabla_omega += self._nabla_omega_inside_expect(nabla_mu_eval, eta)
        return nabla_mu / M, nabla_omega / M
    
    def _approximate_elbo(self, M):
        """Approximate the elbo for the current mu and omega values using MC integration (cf. equation 5)
        [TODO] Do we use the same etas that are sampled during each step of the optimization?
        """
        elbo_left = 0
        for _ in range(M):
            eta = np.random.normal(size=self.dim)
            zeta = self._zeta(eta)
            theta = self._theta(zeta)

            elbo_left += self.model.log_joint(theta) + self.log_jac_inv_T(zeta)
        elbo_left /= M

        entropy = self.omega.sum()  # + constant = 0.5 * self.dim * (1 + np.log(2 * np.π))
        return elbo_left + entropy

    def run(self, learning_rate, M=10, epsilon=0.01, verbose=True):
        # Stochastic optimization
        def get_learning_rate(i, s, grad, tau=1, alpha=0.1):
            s = alpha * grad**2 + (1 - alpha) * s
            rho = learning_rate * (i ** (-0.5 + 1e-16)) / (tau + np.sqrt(s))
            return rho, s
        
        elbo_old = self._approximate_elbo(M)
        delta_elbo = 2 * epsilon

        i = 1
        while np.abs(delta_elbo) > epsilon:  # Change using ELBO
            # Approximate gradients using MC integration
            nabla_mu, nabla_omega = self._gradients_approximate(M)

            # Calculate step size
            if i == 1:
                s_mu, s_omega = nabla_mu ** 2, nabla_omega ** 2
            rho_mu, s_mu = get_learning_rate(i, s_mu, nabla_mu)
            rho_omega, s_omega = get_learning_rate(i, s_omega, nabla_omega)

            # Update mu and omega
            self.update_params(
                self.mu + rho_mu * nabla_mu,
                self.omega + rho_omega * nabla_omega
            )

            # Update elbo
            elbo = self._approximate_elbo(100)
            self.history["elbo"].append(elbo)
            delta_elbo = elbo - elbo_old
            print(i, self.mu)
            elbo_old = elbo

            if verbose and (i % 100 == 0):
                print(f"ELBO {elbo}, mu {self.mu}, omega {self.omega}")

            i += 1


# %%
if __name__ == "__main__":
    N, d = 1000, 1
    betas_true = np.array([5, -3])
    sigma_true = 2.5

    X = np.random.normal(0, 2, (N, d))
    X_ones = np.hstack([np.ones((N, 1)), X])
    y = X_ones @ betas_true + np.random.normal(0, sigma_true, N)

    model = LinearModel(X, y, 0, 10, 1, 2)
    model.describe()

    
    def inv_T(zeta):
        return np.array([*zeta[:-1], np.exp(zeta[-1])], dtype=float)
    inv_T_vec = np.vectorize(inv_T, signature='(n)->(n)')
    
    advi = ADVI(model, inv_T)
    advi.run(learning_rate=0.5)
    print()
    print("Theta True", [*betas_true, sigma_true])
    print("Theta Pred mean", inv_T(advi.mu))
    print("Theta Pred std", inv_T(advi.mu))
# %%

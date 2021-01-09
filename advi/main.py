#%%
from autograd import elementwise_grad, jacobian
import autograd.numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, Math, Latex

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
        # self.history["mu"].append(mu)
        # self.history["omega"].append(omega)

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
            print(i, self.mu[:10])
            # Update elbo
            elbo = self._approximate_elbo(100)
            self.history["elbo"].append(elbo)
            delta_elbo = elbo - elbo_old
            elbo_old = elbo

            if verbose and (i % 10 == 0):
                print(f"ELBO {elbo}, mu {self.mu}, omega {self.omega}")
            i += 1


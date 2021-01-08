#%%
import autograd.numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from autograd import grad, elementwise_grad, jacobian

#%%

class SimpleModel:
    def __init__(self, x, y, ):
        self.x = x
        self.y = y

    def params(self, theta):
        assert len(theta) == 3
        return theta[:2], theta[2]
    
    def _log_priors(self, betas, sigma):
        pass

    def _log_likelihood(self, betas, sigma):
        pass

    def log_joint(self, theta):
        betas, sigma = self.params(theta) 
        return self._log_priors(betas, sigma) + self._log_likelihood(betas, sigma)

#%%

class ADVI:
    def __init__(self, model, inv_T):

        ## Compute Gradients
        self.grad_log_joint = elementwise_grad(model.log_joint)
        self.grad_inv_T = elementwise_grad(inv_T)
        
        jacobian_det_inv_T = lambda zeta: np.linalg.det(jacobian(inv_T))
        self.grad_log_jac_inv_T = elementwise_grad(lambda zeta: np.log(np.abs(jacobian_det_inv_T)))

        # To optimize
        self.mu = np.ones()
        self.omega = np.ones()

    def _nabla_mu_inside_expect(self, eta):
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

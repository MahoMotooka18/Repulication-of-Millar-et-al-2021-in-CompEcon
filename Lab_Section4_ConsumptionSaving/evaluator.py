"""
Evaluation and Metrics Computation

Computes evaluation metrics for assessing solution accuracy:
- Lifetime reward
- Euler equation residuals
"""

import torch
import numpy as np
from typing import Dict, Tuple
from model_consumption_saving import ConsumptionSavingModel


class Evaluator:
    """Computes evaluation metrics on test data."""

    def __init__(self, model: ConsumptionSavingModel, device: str = 'cpu'):
        """
        Initialize evaluator.

        Args:
            model: ConsumptionSavingModel instance
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device

    def evaluate(
        self,
        policy,
        y_test: torch.Tensor,
        w_test: torch.Tensor,
        eps_test: torch.Tensor,
        num_steps: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate policy on test data.

        Args:
            policy: NeuralNetworkPolicy
            y_test: initial log-income (n_test,)
            w_test: initial wealth (n_test,)
            eps_test: shocks (num_steps, n_test)
            num_steps: time horizon for evaluation

        Returns:
            dict with keys:
            - 'lifetime_reward': (n_test,)
            - 'euler_residual_a': (num_steps, n_test)
            - 'euler_residual_h': (num_steps, n_test)
            - 'euler_residual_fb': (num_steps, n_test)
        """
        n_test = y_test.shape[0]

        lifetime_rewards = np.zeros(n_test)
        euler_residual_a = np.zeros((num_steps, n_test))
        euler_residual_h = np.zeros((num_steps, n_test))
        euler_residual_fb = np.zeros((num_steps, n_test))

        y_t = y_test.clone()
        w_t = w_test.clone()
        gh_nodes, gh_weights = self.model.create_gauss_hermite_quadrature(10)
        gh_nodes_t = torch.from_numpy(gh_nodes).to(self.device)
        gh_weights_t = torch.from_numpy(gh_weights).to(self.device)

        with torch.no_grad():
            for t in range(num_steps):
                # Get consumption
                c_t = w_t * policy.forward_phi(y_t, w_t)
                c_t = torch.clamp(c_t, min=torch.zeros_like(w_t), max=w_t)

                # Utility at current consumption
                u_c_t = self._utility_derivative_torch(c_t)

                # Add to lifetime reward
                u_t = self._utility_torch(c_t)
                discount = self.model.params.beta ** t
                lifetime_rewards += (discount * u_t.cpu().numpy())

                x = eps_test[t] if t < eps_test.shape[0] else np.zeros(n_test)

                if isinstance(x, torch.Tensor):
                    eps_t = x.to(self.device).float()
                else:
                # numpy array (or array-like)
                    eps_t = torch.as_tensor(x, device=self.device, dtype=torch.float32)

                y_next = self.model.income_transition(y_t, eps_t)
                w_next = self.model.state_transition(w_t, c_t, y_t)

                # Expected marginal utility via Gauss-Hermite quadrature
                u_c_next_exp = torch.zeros_like(u_c_t)
                for node, weight in zip(gh_nodes_t, gh_weights_t):
                    y_next_gh = (self.model.params.rho * y_t +
                                 self.model.params.sigma * node)
                    c_next = w_next * policy.forward_phi(y_next_gh, w_next)
                    c_next = torch.clamp(
                        c_next,
                        min=torch.zeros_like(w_next),
                        max=w_next
                    )
                    u_c_next = self._utility_derivative_torch(c_next)
                    u_c_next_exp += weight * u_c_next

                # Euler residuals
                a = 1.0 - c_t / w_t
                h = 1.0 - (
                    self.model.params.beta * self.model.params.r *
                    u_c_next_exp / u_c_t
                )
                fb = a + h - torch.sqrt(a ** 2 + h ** 2 + 1e-12)

                euler_residual_a[t] = a.cpu().numpy()
                euler_residual_h[t] = h.cpu().numpy()
                euler_residual_fb[t] = fb.cpu().numpy()

                # Transition
                y_t = y_next
                w_t = w_next

        return {
            'lifetime_reward': lifetime_rewards,
            'euler_residual_a': euler_residual_a,
            'euler_residual_h': euler_residual_h,
            'euler_residual_fb': euler_residual_fb,
        }

    def _utility_torch(self, c: torch.Tensor) -> torch.Tensor:
        """CRRA utility on torch tensors."""
        gamma = self.model.params.gamma
        if gamma == 1.0:
            return torch.log(c)
        return (c ** (1 - gamma) - 1) / (1 - gamma)

    def _utility_derivative_torch(self, c: torch.Tensor) -> torch.Tensor:
        """Marginal utility on torch tensors."""
        gamma = self.model.params.gamma
        return c ** (-gamma)

    def compute_statistics(
        self,
        metrics: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute summary statistics from evaluation metrics.

        Args:
            metrics: output from evaluate()

        Returns:
            dict with statistics
        """
        stats = {}

        # Lifetime reward statistics
        lr = metrics['lifetime_reward']
        stats['lifetime_reward_mean'] = float(np.mean(lr))
        stats['lifetime_reward_std'] = float(np.std(lr))

        # Euler residual FB statistics
        fb = metrics['euler_residual_fb']
        abs_fb = np.abs(fb)
        stats['euler_fb_mean'] = float(np.mean(abs_fb))
        stats['euler_fb_p50'] = float(np.percentile(abs_fb, 50))
        stats['euler_fb_p90'] = float(np.percentile(abs_fb, 90))
        stats['euler_fb_max'] = float(np.max(abs_fb))

        # Euler residual magnitude
        euler_magnitude = np.sqrt(
            metrics['euler_residual_a'] ** 2 +
            metrics['euler_residual_h'] ** 2
        )
        stats['euler_magnitude_mean'] = float(np.mean(euler_magnitude))
        stats['euler_magnitude_p50'] = float(np.percentile(euler_magnitude, 50))
        stats['euler_magnitude_p90'] = float(np.percentile(euler_magnitude, 90))

        return stats

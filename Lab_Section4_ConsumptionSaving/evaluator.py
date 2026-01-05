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
        num_steps: int = 100,
        violation_sample_limit: int = 0
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
        violation_count = 0
        violation_samples = []
        eps = 1e-12

        y_t = y_test.clone()
        w_t = w_test.clone()
        gh_nodes, gh_weights = self.model.create_gauss_hermite_quadrature(10)
        gh_nodes_t = torch.from_numpy(gh_nodes).to(self.device)
        gh_weights_t = torch.from_numpy(gh_weights).to(self.device)

        with torch.no_grad():
            for t in range(num_steps):
                # Get consumption with domain guard
                c_raw = w_t * policy.forward_phi(y_t, w_t)
                c_violation = (~torch.isfinite(c_raw) |
                               ~torch.isfinite(w_t) |
                               (c_raw <= 0.0) |
                               (w_t <= 0.0))
                if torch.any(c_violation):
                    violation_count += int(torch.sum(c_violation).item())
                    if violation_sample_limit > 0 and len(violation_samples) < violation_sample_limit:
                        idxs = torch.nonzero(c_violation).flatten()
                        for idx in idxs[:(violation_sample_limit - len(violation_samples))]:
                            i = int(idx.item())
                            violation_samples.append({
                                't': int(t),
                                'c_raw': float(c_raw[i].item()),
                                'w_t': float(w_t[i].item()),
                                'c_nonpositive': bool(c_raw[i].item() <= 0.0),
                                'w_nonpositive': bool(w_t[i].item() <= 0.0),
                                'c_nonfinite': bool(not torch.isfinite(c_raw[i]).item()),
                                'w_nonfinite': bool(not torch.isfinite(w_t[i]).item())
                            })

                positive_w = w_t > 0.0
                c_t = torch.where(
                    positive_w,
                    torch.clamp(c_raw, min=torch.zeros_like(w_t), max=w_t),
                    torch.zeros_like(w_t)
                )

                # Utility at current consumption
                c_safe = torch.clamp(c_t, min=eps)
                w_safe = torch.clamp(w_t, min=eps)
                u_c_t = self._utility_derivative_torch(c_safe)

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
                    c_next_raw = w_next * policy.forward_phi(y_next_gh, w_next)
                    positive_w_next = w_next > 0.0
                    c_next = torch.where(
                        positive_w_next,
                        torch.clamp(
                            c_next_raw,
                            min=torch.zeros_like(w_next),
                            max=w_next
                        ),
                        torch.zeros_like(w_next)
                    )
                    c_next_safe = torch.clamp(c_next, min=eps)
                    u_c_next = self._utility_derivative_torch(c_next_safe)
                    u_c_next_exp += weight * u_c_next

                # Euler residuals
                a = 1.0 - c_t / w_safe
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
            'violation_count': violation_count,
            'violation_samples': violation_samples,
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

        # Lifetime reward statistics with finite guard
        lr = metrics['lifetime_reward']
        lr_finite = np.isfinite(lr)
        if np.any(lr_finite):
            stats['lifetime_reward_mean'] = float(np.mean(lr[lr_finite]))
        else:
            stats['lifetime_reward_mean'] = float('nan')

        # Euler residual FB statistics with finite guard
        fb = metrics['euler_residual_fb']
        abs_fb = np.abs(fb)
        fb_finite = np.isfinite(abs_fb)
        if abs_fb.size == 0:
            stats['euler_fb_finite_ratio'] = 0.0
            stats['euler_fb_mean'] = float('nan')
        else:
            stats['euler_fb_finite_ratio'] = float(np.mean(fb_finite))
            if np.any(fb_finite):
                stats['euler_fb_mean'] = float(np.mean(abs_fb[fb_finite]))
            else:
                stats['euler_fb_mean'] = float('nan')

        return stats

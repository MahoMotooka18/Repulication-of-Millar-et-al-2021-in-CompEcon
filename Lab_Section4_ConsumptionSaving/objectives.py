"""
Objective Functions for Deep Learning Training

Implements three training objectives:
1. Lifetime Reward: maximize discounted utility
2. Euler Equation: minimize Euler residuals with complementarity
3. Bellman Equation: minimize Bellman residuals
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict
import numpy as np
from model_consumption_saving import ConsumptionSavingModel


class ObjectiveComputer:
    """Computes loss objectives for the three training methods."""

    def __init__(self, model: ConsumptionSavingModel, device: str = 'cpu'):
        """
        Initialize objective computer.

        Args:
            model: ConsumptionSavingModel instance
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device

    # =====================================================================
    # Lifetime Reward Objective (Eq. 27)
    # =====================================================================

    def lifetime_reward_objective(
        self,
        y_path: torch.Tensor,
        w_path: torch.Tensor,
        c_path: torch.Tensor
    ) -> torch.Tensor:
        """
        Lifetime Reward objective (Eq. 27).

        L^LR(theta) = -sum_t beta^t * u(c_t)
        (negative because we minimize loss, not maximize reward)

        Args:
            y_path: log-income path (T, batch_size)
            w_path: wealth path (T, batch_size)
            c_path: consumption path (T, batch_size)

        Returns:
            scalar loss (negative of lifetime reward)
        """
        T = c_path.shape[0]
        batch_size = c_path.shape[1]

        # Compute utilities
        utilities = self._utility_batch(c_path)  # (T, batch_size)

        # Apply discount factors and sum
        discount_factors = torch.tensor(
            [self.model.params.beta ** t for t in range(T)],
            device=self.device,
            dtype=torch.float32
        ).view(-1, 1)

        lifetime_reward = torch.sum(discount_factors * utilities, dim=0)
        loss = -lifetime_reward.mean()  # Negate: minimize means maximize LR

        return loss

    # =====================================================================
    # Euler Equation Objective (Eq. 28-30)
    # =====================================================================

    def euler_objective(
        self,
        policy,
        y_batch: torch.Tensor,
        w_batch: torch.Tensor,
        eps1_batch: torch.Tensor,
        eps2_batch: torch.Tensor,
        nu_h: float = 1.0,
        nu: float = 1.0
    ) -> torch.Tensor:
        """
        Euler equation objective (Eq. 30) with AiO and two uncorrelated shocks.

        Uses method of two uncorrelated shocks to approximate expectations.

        Args:
            policy: NeuralNetworkPolicy (must have forward_phi, forward_h)
            y_batch: current log-income (n_samples,)
            w_batch: current wealth (n_samples,)
            eps1_batch: first shock (n_samples,)
            eps2_batch: second shock (n_samples,)
            nu_h: weight on multiplier matching term
            nu: weight on FB residual term

        Returns:
            scalar loss (Eq. 30)
        """
        batch_size = y_batch.shape[0]

        # Current state consumption and multiplier
        c = w_batch * policy.forward_phi(y_batch, w_batch)
        h = policy.forward_h(y_batch, w_batch)

        # Transition for shock 1
        y_next_1 = self.model.income_transition(y_batch, eps1_batch)
        w_next_1 = self.model.state_transition(w_batch, c, y_batch)
        c_next_1 = w_next_1 * policy.forward_phi(y_next_1, w_next_1)

        # Transition for shock 2
        y_next_2 = self.model.income_transition(y_batch, eps2_batch)
        w_next_2 = self.model.state_transition(w_batch, c, y_batch)
        c_next_2 = w_next_2 * policy.forward_phi(y_next_2, w_next_2)

        # Compute marginal utilities
        u_c = self._utility_derivative_batch(c)
        u_c_next_1 = self._utility_derivative_batch(c_next_1)
        u_c_next_2 = self._utility_derivative_batch(c_next_2)

        # FB slackness term
        a = 1.0 - c / w_batch

        # FB residual
        fb_residual = self._fischer_burmeister_batch(a, 1.0 - h)

        # Euler expectation terms
        euler_1 = (self.model.params.beta * self.model.params.r *
                   u_c_next_1 / u_c - h)
        euler_2 = (self.model.params.beta * self.model.params.r *
                   u_c_next_2 / u_c - h)

        # Loss (Eq. 30)
        loss_fb = torch.mean(fb_residual ** 2)
        loss_euler = torch.mean(euler_1 * euler_2)

        total_loss = loss_fb + nu_h * loss_euler

        return total_loss

    # =====================================================================
    # Bellman Equation Objective (Eq. 31-32)
    # =====================================================================

    def bellman_objective(
        self,
        policy,
        y_batch: torch.Tensor,
        w_batch: torch.Tensor,
        eps1_batch: torch.Tensor,
        eps2_batch: torch.Tensor,
        nu_h: float = 1.0,
        nu: float = 1.0
    ) -> torch.Tensor:
        """
        Bellman equation objective (Eq. 32) with AiO and two shocks.

        Combines Bellman residual with FB characterization of optimality.

        Args:
            policy: NeuralNetworkPolicy (must have all three outputs)
            y_batch: current log-income (n_samples,)
            w_batch: current wealth (n_samples,)
            eps1_batch: first shock (n_samples,)
            eps2_batch: second shock (n_samples,)
            nu_h: weight on multiplier matching
            nu: weight on FB residual term

        Returns:
            scalar loss (Eq. 32)
        """
        batch_size = y_batch.shape[0]

        # Current state values
        phi = policy.forward_phi(y_batch, w_batch)
        c = w_batch * phi
        h = policy.forward_h(y_batch, w_batch)
        V = policy.forward_v(y_batch, w_batch)

        # Transition for shock 1
        y_next_1 = self.model.income_transition(y_batch, eps1_batch)
        w_next_1 = self.model.state_transition(w_batch, c, y_batch)
        V_next_1 = policy.forward_v(y_next_1, w_next_1)
        c_next_1 = w_next_1 * policy.forward_phi(y_next_1, w_next_1)

        # Transition for shock 2
        y_next_2 = self.model.income_transition(y_batch, eps2_batch)
        w_next_2 = self.model.state_transition(w_batch, c, y_batch)
        V_next_2 = policy.forward_v(y_next_2, w_next_2)
        c_next_2 = w_next_2 * policy.forward_phi(y_next_2, w_next_2)

        # Utility at current consumption
        u_c = self._utility_batch(c)

        # Bellman residual for two shocks
        bellman_1 = V - u_c - self.model.params.beta * V_next_1
        bellman_2 = V - u_c - self.model.params.beta * V_next_2
        loss_bellman = torch.mean(bellman_1 * bellman_2)

        # Value derivatives (approximated via finite differences)
        dV_dw_1 = self._value_derivative_w(
            policy, y_next_1, w_next_1, h_value=1e-4
        )
        dV_dw_2 = self._value_derivative_w(
            policy, y_next_2, w_next_2, h_value=1e-4
        )

        # Marginal utilities
        u_c_next_1 = self._utility_derivative_batch(c_next_1)
        u_c_next_2 = self._utility_derivative_batch(c_next_2)

        # FB slackness and multiplier terms
        a = 1.0 - c / w_batch
        fb_residual = self._fischer_burmeister_batch(a, 1.0 - h)

        # Multiplier expectation matching
        mult_1 = (self.model.params.beta * dV_dw_1 / u_c - h)
        mult_2 = (self.model.params.beta * dV_dw_2 / u_c - h)

        loss_fb = torch.mean(fb_residual ** 2)
        loss_mult = torch.mean(mult_1 * mult_2)

        total_loss = (loss_bellman + nu * loss_fb + nu_h * loss_mult)

        return total_loss

    # =====================================================================
    # Helper Methods
    # =====================================================================

    def _utility_batch(self, c: torch.Tensor) -> torch.Tensor:
        """
        CRRA utility (batched).

        u(c) = (c^(1-gamma) - 1) / (1 - gamma)
        """
        gamma = self.model.params.gamma
        if gamma == 1.0:
            return torch.log(c)
        else:
            return (c ** (1 - gamma) - 1) / (1 - gamma)

    def _utility_derivative_batch(self, c: torch.Tensor) -> torch.Tensor:
        """Marginal utility u'(c) = c^(-gamma)."""
        gamma = self.model.params.gamma
        return c ** (-gamma)

    def _fischer_burmeister_batch(
        self,
        a: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        Fischer-Burmeister function (batched).

        Psi^FB(a, h) = a + h - sqrt(a^2 + h^2)
        """
        return a + h - torch.sqrt(a ** 2 + h ** 2 + 1e-12)

    def _value_derivative_w(
        self,
        policy,
        y: torch.Tensor,
        w: torch.Tensor,
        h_value: float = 1e-4
    ) -> torch.Tensor:
        """
        Approximate dV/dw using finite differences.

        dV/dw ≈ (V(y, w+h) - V(y, w-h)) / (2*h)
        """
        w_plus = w + h_value
        w_minus = w - h_value

        V_plus = policy.forward_v(y, w_plus)
        V_minus = policy.forward_v(y, w_minus)

        dV_dw = (V_plus - V_minus) / (2 * h_value)
        return dV_dw

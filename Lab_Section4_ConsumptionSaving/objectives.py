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
        Bellman equation objective (Eq. 32 in paper) with AiO.

        Implements the objective from Definition 2.10 in the paper:
        
        Ξ(θ) = E_ω[ξ(ω;θ)] 
        
        = E_{(y,w,ε₁,ε₂)} { [V(y,w;θ) - u(c) - βV(y',w';θ)|_{ε=ε₁}]
                              × [V(y,w;θ) - u(c) - βV(y',w';θ)|_{ε=ε₂}]
                              + ν·[Ψ^FB(1-c/w, 1-h)]²
                              + ν_h·[(β·∂V/∂w'|_{ε₁}/u'(c) - h)
                                    ·(β·∂V/∂w'|_{ε₂}/u'(c) - h)] }

        Key components:
        1. Bellman residual (AiO product): squared residuals via two independent shocks
        2. Fischer-Burmeister complementarity: ensures a·h = 0 and a,h ≥ 0
        3. Multiplier matching: enforces first-order conditions across shocks

        Args:
            policy: NeuralNetworkPolicy with forward_phi, forward_h, forward_v
            y_batch: current log-income (n_samples,)
            w_batch: current wealth (n_samples,)
            eps1_batch: first shock (n_samples,) — independent
            eps2_batch: second shock (n_samples,) — independent from eps1
            nu: weight on Fischer-Burmeister term (default 1.0)
            nu_h: weight on multiplier matching term (default 1.0)

        Returns:
            scalar loss to minimize (Eq. 32)
        """
        batch_size = y_batch.shape[0]

        # ===== Current state =====
        phi = policy.forward_phi(y_batch, w_batch)
        c = w_batch * phi
        c = torch.clamp(c, min=torch.zeros_like(c), max=w_batch)  # Ensure feasibility
        V = policy.forward_v(y_batch, w_batch)
        u_c = self._utility_batch(c)  # Utility value
        u_c_deriv = self._utility_derivative_batch(c)  # Marginal utility

        # ===== Shock 1: First AiO term =====
        y_next_1 = self.model.income_transition(y_batch, eps1_batch)
        w_next_1 = self.model.state_transition(w_batch, c, y_batch)  # Use current y
        V_next_1 = policy.forward_v(y_next_1, w_next_1)

        # ===== Shock 2: Second AiO term (independent) =====
        y_next_2 = self.model.income_transition(y_batch, eps2_batch)
        w_next_2 = self.model.state_transition(w_batch, c, y_batch)  # Use current y (not y_next_1)
        V_next_2 = policy.forward_v(y_next_2, w_next_2)

        # ===== Bellman residuals for AiO product =====
        # R(y,w,ε) = V(y,w) - u(c) - β·V(y',w')
        bellman_1 = V - u_c - self.model.params.beta * V_next_1
        bellman_2 = V - u_c - self.model.params.beta * V_next_2
        loss_bellman = torch.mean(bellman_1 * bellman_2)

        # ===== Value function gradient (for FOC constraint) =====
        dV_dw_1 = self._value_derivative_w(policy, y_next_1, w_next_1, h_value=1e-4)
        dV_dw_2 = self._value_derivative_w(policy, y_next_2, w_next_2, h_value=1e-4)

        # ===== Complementary slackness: a·λ = 0 with a,λ ≥ 0 =====
        eps_guard = 1e-10
        a = 1.0 - c / (w_batch + eps_guard)
        a = torch.clamp(a, min=0.0)
        
        # Lagrange multiplier from FOC: λ = 1 - (β·r·∂V/∂w')/u'(c)
        lambda_1 = 1.0 - (self.model.params.beta * self.model.params.r * dV_dw_1) / (u_c_deriv + eps_guard)
        lambda_2 = 1.0 - (self.model.params.beta * self.model.params.r * dV_dw_2) / (u_c_deriv + eps_guard)
        lambda_1 = torch.clamp(lambda_1, min=0.0)
        lambda_2 = torch.clamp(lambda_2, min=0.0)
        
        # Fischer-Burmeister: Ψ^FB(a,λ) = a + λ - √(a² + λ²)
        fb_1 = self._fischer_burmeister_batch(a, lambda_1)
        fb_2 = self._fischer_burmeister_batch(a, lambda_2)
        loss_fb = torch.mean(fb_1 * fb_2)

        # ===== Multiplier consistency across shocks (AiO) =====
        # Ensure λ satisfies FOC under both shock realizations
        mult_1 = a * lambda_1
        mult_2 = a * lambda_2
        loss_mult = torch.mean(mult_1 * mult_2)

        # ===== Combined loss (Eq. 32) =====
        total_loss = loss_bellman + nu * loss_fb + nu_h * loss_mult
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
        Approximate dV/dw using one-sided finite differences.
        
        Uses forward difference to avoid negative w values:
        dV/dw ≈ (V(y, w+h) - V(y, w)) / h
        
        This ensures w remains in the valid domain [0, ∞).
        """
        # Ensure w is positive
        w_safe = torch.clamp(w, min=1e-10)
        w_plus = w_safe + h_value

        V_base = policy.forward_v(y, w_safe)
        V_plus = policy.forward_v(y, w_plus)

        dV_dw = (V_plus - V_base) / h_value
        
        # Guard against non-finite gradients
        dV_dw = torch.where(
            torch.isfinite(dV_dw),
            dV_dw,
            torch.zeros_like(dV_dw)
        )
        
        return dV_dw

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
    """
    Computes three training objectives for Section 4 (Maliar et al. 2021).
    
    Implements the three fundamental methods for solving dynamic economic models:
    1. Lifetime Reward: Direct maximization of expected discounted utility
    2. Euler Equation: Minimizes residuals in optimality conditions + complementarity
    3. Bellman Equation: Minimizes value function and FOC residuals
    
    All objectives use the AiO (All-in-One) operator with two independent shocks
    to reduce the dimensionality of expectation computations.
    """

    def __init__(self, model: ConsumptionSavingModel, device: str = 'cpu') -> None:
        """
        Initialize objective computer.

        Args:
            model: ConsumptionSavingModel instance with parameters and utilities.
            device: 'cpu' or 'cuda' for tensor placement.
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
        Compute Lifetime Reward objective (Section 2.3.5 of note.md, Eq. 27).

        The lifetime reward is the discounted sum of utilities along a simulated path:
            E[LR] = E_{(y₀,w₀,ε₁,...,εₜ)} [Σₜ βᵗ u(cₜ)]
        
        For training, we maximize by minimizing the negative:
            Loss = -E[LR] = -E[Σₜ βᵗ u(cₜ)]

        Args:
            y_path: Log-income path of shape (T, batch_size).
            w_path: Cash-on-hand path of shape (T, batch_size).
            c_path: Consumption path of shape (T, batch_size).

        Returns:
            Scalar loss tensor (mean negative lifetime reward).
        """
        T = c_path.shape[0]
        batch_size = c_path.shape[1]

        # Compute utilities along path
        utilities = self._utility_batch(c_path)  # (T, batch_size)

        # Create discount factor vector β^t for t = 0, 1, ..., T-1
        discount_factors = torch.tensor(
            [self.model.params.beta ** t for t in range(T)],
            device=self.device,
            dtype=torch.float32
        ).view(-1, 1)  # (T, 1)

        # Compute lifetime reward: sum_t β^t * u(c_t)
        lifetime_reward = torch.sum(discount_factors * utilities, dim=0)  # (batch_size,)
        
        # Return negative (to minimize) averaged over batch
        loss = -lifetime_reward.mean()

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
        Compute Euler equation objective (Section 2.3.6 of note.md, Eq. 30).

        Minimizes the Kuhn-Tucker conditions for the borrowing-constrained problem:
            a ≥ 0, h ≥ 0, a·h = 0
        
        where:
            a = 1 - c/w  (complementary slackness)
            h = 1 - (β*r*u'(c'))/u'(c)  (Euler residual)
        
        Encodes via Fischer-Burmeister: Ψ^FB(a,h) = a + h - √(a² + h²)
        
        Uses AiO operator with two independent shocks ε₁, ε₂:
            Loss = E[Ψ^FB(a,1-h)²] + νₕ·E[(βru'(c')|ε₁/u'(c) - h)
                                          ·(βru'(c')|ε₂/u'(c) - h)]

        Args:
            policy: NeuralNetworkPolicy with forward_phi and forward_h methods.
            y_batch: Current log-income, shape (n_samples,).
            w_batch: Current cash-on-hand, shape (n_samples,).
            eps1_batch: First shock (independent), shape (n_samples,).
            eps2_batch: Second shock (independent), shape (n_samples,).
            nu_h: Weight on multiplier consistency term (default 1.0).
            nu: Weight on Fischer-Burmeister term (default 1.0).

        Returns:
            Scalar loss tensor.
        """
        batch_size = y_batch.shape[0]

        # ===== Current state =====
        # Consumption from policy: c = w * phi(y, w)
        phi = policy.forward_phi(y_batch, w_batch)
        c = w_batch * phi
        
        # Lagrange multiplier estimate
        h = policy.forward_h(y_batch, w_batch)

        # ===== Shock 1: Next state and consumption =====
        y_next_1 = self.model.income_transition(y_batch, eps1_batch)
        w_next_1 = self.model.state_transition(w_batch, c, y_batch)
        c_next_1 = w_next_1 * policy.forward_phi(y_next_1, w_next_1)

        # ===== Shock 2: Next state and consumption (independent) =====
        y_next_2 = self.model.income_transition(y_batch, eps2_batch)
        w_next_2 = self.model.state_transition(w_batch, c, y_batch)  # Same c, different y_next
        c_next_2 = w_next_2 * policy.forward_phi(y_next_2, w_next_2)

        # ===== Marginal utilities =====
        u_c = self._utility_derivative_batch(c)
        u_c_next_1 = self._utility_derivative_batch(c_next_1)
        u_c_next_2 = self._utility_derivative_batch(c_next_2)

        # ===== Complementary slackness term =====
        # a = 1 - c/w
        a = 1.0 - c / w_batch

        # ===== Fischer-Burmeister complementarity =====
        # Ψ^FB(a, 1 - h) = a + (1-h) - √(a² + (1-h)²)
        fb_residual = self._fischer_burmeister_batch(a, 1.0 - h)
        loss_fb = torch.mean(fb_residual ** 2)

        # ===== Euler residuals (AiO product) =====
        # Euler residual: βr·u'(c')/u'(c) - h
        euler_1 = (self.model.params.beta * self.model.params.r *
                   u_c_next_1 / u_c - h)
        euler_2 = (self.model.params.beta * self.model.params.r *
                   u_c_next_2 / u_c - h)
        loss_euler = torch.mean(euler_1 * euler_2)

        # ===== Combined loss (Eq. 30) =====
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
        Compute Bellman equation objective (Section 2.3.7 of note.md, Eq. 32).

        Combines three terms from the Bellman approach:
        1. Bellman residual (value function accuracy)
        2. Fischer-Burmeister complementarity (optimality conditions)
        3. Multiplier consistency (FOC enforcement across shocks)
        
        The Bellman equation for the borrowing-constrained problem is:
            V(y,w) = max_c { u(c) + β·E[V(y',w')] }
        subject to 0 ≤ c ≤ w
        
        This is encoded as a minimization of residuals via AiO (two shocks).

        Args:
            policy: NeuralNetworkPolicy with forward_phi, forward_h, forward_v.
            y_batch: Current log-income, shape (n_samples,).
            w_batch: Current cash-on-hand, shape (n_samples,).
            eps1_batch: First shock (independent), shape (n_samples,).
            eps2_batch: Second shock (independent), shape (n_samples,).
            nu: Weight on Fischer-Burmeister term (default 1.0).
            nu_h: Weight on multiplier consistency term (default 1.0).

        Returns:
            Scalar loss tensor to minimize.
        """
        batch_size = y_batch.shape[0]
        eps_guard = 1e-10  # Guard against division by zero

        # ===== Current state =====
        phi = policy.forward_phi(y_batch, w_batch)
        c = w_batch * phi
        c = torch.clamp(c, min=0.0, max=w_batch)  # Enforce feasibility
        
        V = policy.forward_v(y_batch, w_batch)
        u_c = self._utility_batch(c)  # Utility value
        u_c_deriv = self._utility_derivative_batch(c)  # Marginal utility

        # ===== AiO Shock 1 =====
        y_next_1 = self.model.income_transition(y_batch, eps1_batch)
        w_next_1 = self.model.state_transition(w_batch, c, y_batch)
        V_next_1 = policy.forward_v(y_next_1, w_next_1)
        dV_dw_1 = self._value_derivative_w(policy, y_next_1, w_next_1, h_value=1e-4)

        # ===== AiO Shock 2 (independent) =====
        y_next_2 = self.model.income_transition(y_batch, eps2_batch)
        w_next_2 = self.model.state_transition(w_batch, c, y_batch)  # Same c as shock 1
        V_next_2 = policy.forward_v(y_next_2, w_next_2)
        dV_dw_2 = self._value_derivative_w(policy, y_next_2, w_next_2, h_value=1e-4)

        # ===== Bellman residuals (AiO product, Eq. 32 first term) =====
        # R(y,w,ε) = V(y,w) - u(c) - β·V(y',w')
        bellman_1 = V - u_c - self.model.params.beta * V_next_1
        bellman_2 = V - u_c - self.model.params.beta * V_next_2
        loss_bellman = torch.mean(bellman_1 * bellman_2)

        # ===== Complementary slackness (slackness variable) =====
        # a = 1 - c/w (savings = w - c, a = savings/w)
        a = 1.0 - c / (w_batch + eps_guard)
        a = torch.clamp(a, min=0.0)

        # ===== Policy network multiplier estimate (for FB term) =====
        # h from policy network (Eq. 32 second term: Ψ^FB(a, 1-h)²)
        h = policy.forward_h(y_batch, w_batch)

        # ===== Fischer-Burmeister complementarity (Eq. 32 second term) =====
        # Ψ^FB(a, 1-h) = a + (1-h) - √(a² + (1-h)²)
        # Computed once with policy h, then squared (NOT product of two independent computations)
        fb = self._fischer_burmeister_batch(a, 1.0 - h)
        loss_fb = torch.mean(fb ** 2)

        # ===== Lagrange multiplier from FOC (for multiplier consistency term) =====
        # λ = 1 - (β·r·∂V/∂w')/u'(c)
        # This represents the KKT multiplier on the borrowing constraint, computed from value function
        h_1 = 1.0 - (self.model.params.beta * self.model.params.r * dV_dw_1) / (u_c_deriv + eps_guard)
        h_2 = 1.0 - (self.model.params.beta * self.model.params.r * dV_dw_2) / (u_c_deriv + eps_guard)
        h_1 = torch.clamp(h_1, min=0.0)
        h_2 = torch.clamp(h_2, min=0.0)

        # ===== Multiplier consistency (AiO, Eq. 32 third term) =====
        # Enforce h·a ≈ 0 under both shock realizations
        mult_1 = a * h_1
        mult_2 = a * h_2
        loss_mult = torch.mean(mult_1 * mult_2)

        # ===== Combined loss (Eq. 32) =====
        total_loss = loss_bellman + nu * loss_fb + nu_h * loss_mult
        return total_loss

    # =====================================================================
    # Helper Methods
    # =====================================================================

    def _utility_batch(self, c: torch.Tensor) -> torch.Tensor:
        """
        Compute CRRA utility for a batch of consumption values.

        Utility function: u(c) = (c^(1-γ) - 1) / (1 - γ)
        Special case: u(c) = ln(c) when γ = 1.0

        Args:
            c: Consumption tensor of any shape.

        Returns:
            Utility tensor with same shape as input.
        """
        gamma = self.model.params.gamma
        if gamma == 1.0:
            return torch.log(c)
        else:
            return (c ** (1 - gamma) - 1) / (1 - gamma)

    def _utility_derivative_batch(self, c: torch.Tensor) -> torch.Tensor:
        """
        Compute marginal utility for a batch of consumption values.

        Marginal utility: u'(c) = c^(-γ)

        Args:
            c: Consumption tensor of any shape.

        Returns:
            Marginal utility tensor with same shape as input.
        """
        gamma = self.model.params.gamma
        return c ** (-gamma)

    def _fischer_burmeister_batch(
        self,
        a: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate Fischer-Burmeister complementarity function (batched).

        Fischer-Burmeister function:
            Ψ^FB(a, h) = a + h - √(a² + h²)
        
        This encodes complementary slackness conditions: a·h = 0, a ≥ 0, h ≥ 0.
        It is zero iff all three conditions hold.

        Args:
            a: Slackness term tensor.
            h: Lagrange multiplier term tensor.

        Returns:
            Fischer-Burmeister residual tensor with same shape.
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
        Approximate ∂V/∂w using one-sided finite differences.

        Uses forward difference to keep w in valid domain [0, ∞):
            ∂V/∂w ≈ [V(y, w+h) - V(y, w)] / h
        
        This derivative is used in the FOC constraint for the Bellman method
        to encode the optimality condition:
            u'(c) = β·r·∂V/∂w'
        
        Args:
            policy: NeuralNetworkPolicy with forward_v method.
            y: Log-income, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
            h_value: Finite difference step size (default 1e-4).

        Returns:
            Approximate ∂V/∂w, shape (batch_size,).
        """
        # Ensure w is positive to avoid numerical issues
        w_safe = torch.clamp(w, min=1e-10)
        w_plus = w_safe + h_value

        # Evaluate V at two points
        V_base = policy.forward_v(y, w_safe)
        V_plus = policy.forward_v(y, w_plus)

        # Finite difference
        dV_dw = (V_plus - V_base) / h_value
        
        # Guard against non-finite values
        dV_dw = torch.where(
            torch.isfinite(dV_dw),
            dV_dw,
            torch.zeros_like(dV_dw)
        )
        
        return dV_dw

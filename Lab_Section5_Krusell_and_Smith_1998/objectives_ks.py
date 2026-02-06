"""
Objective Functions for Krusell-Smith Model

Implements three training objectives:
1. Lifetime Reward
2. Euler Equation with complementarity
3. Bellman Equation
"""

import torch
import numpy as np
from typing import Tuple
from model_ks1998 import KrusellSmithModel
from policy_utils_ks import InputScaleSpec, consumption_from_share_torch


class KSObjectiveComputer:
    """
    Computes loss objectives for training Krusell-Smith heterogeneous-agent models.
    
    Implements three training objectives from Maliar et al. (2021) Algorithm 1:
    1. Lifetime Reward: Maximizes discounted expected utility (paper Eq. 27)
    2. Euler Equation: Enforces intertemporal consumption trade-off with 
       Fischer-Burmeister complementarity (paper Eq. 44)
    3. Bellman Equation: Matches value function via dynamic programming equation
       (paper Eq. 32, adapted for heterogeneous agents)
    
    For heterogeneous agents, each objective is computed agent-by-agent within
    the batch, aggregating via cross-sectional means. Distribution features D_t
    are passed to neural network to ensure agents respond to economy-wide conditions.
    """

    def __init__(
        self,
        model: KrusellSmithModel,
        device: str = 'cpu'
    ) -> None:
        """
        Initialize Krusell-Smith objective computer.
        
        Args:
            model: KrusellSmithModel instance with parameters (gamma, beta, alpha, delta, etc.)
                and methods for utility computation and Euler residuals.
            device: Computing device ('cpu' or 'cuda'). Determines where tensors are allocated.
        """
        self.model = model
        self.device = device

    # =====================================================================
    # Lifetime Reward Objective
    # =====================================================================

    def lifetime_reward_objective(
        self,
        c_path: torch.Tensor,
        w_path: torch.Tensor
    ) -> torch.Tensor:
        """
        Lifetime Reward objective (Maliar et al. 2021, Eq. 27).
        
        Maximizes V^LR = E[∑_t β^t u(c_t)] where u(c) is CRRA utility.
        This objective directly optimizes discounted lifetime utility and serves
        as a baseline training target. Works for all agent/batch combinations within
        the heterogeneous-agent framework.
        
        Formula:
            u(c) = (c^(1-γ) - 1) / (1-γ)  if γ ≠ 1
            u(c) = log(c)                   if γ = 1
        
        Args:
            c_path: Consumption path indexed by (T, batch_size, num_agents).
                T is time horizon, batch_size is gradient batch, num_agents is 
                heterogeneous agents in the economy.
            w_path: Wealth path (not directly used in utility, kept for API consistency).
                Shape (T, batch_size, num_agents).
        
        Returns:
            torch.Tensor: Scalar loss (negative of mean lifetime reward).
                Returns the negative reward since optimization minimizes loss.
        """
        T = c_path.shape[0]
        gamma = self.model.params.gamma
        
        # Compute utilities for all time periods
        if abs(gamma - 1.0) < 1e-10:
            utilities = torch.log(c_path + 1e-8)
        else:
            utilities = (c_path**(1 - gamma) - 1) / (1 - gamma)
        
        # Create discount factors β^t for t = 0, 1, ..., T-1
        discount_factors = torch.tensor(
            [self.model.params.beta**t for t in range(T)],
            device=self.device,
            dtype=torch.float32
        ).view(-1, 1, 1)  # Shape: (T, 1, 1) for broadcasting
        
        # Compute lifetime reward: ∑_t β^t u(c_t)
        lifetime_reward = torch.sum(discount_factors * utilities, dim=0)
        
        # Return negative mean as loss (minimize negative reward = maximize reward)
        loss = -lifetime_reward.mean()
        
        return loss

    # =====================================================================
    # Euler Equation Objective
    # =====================================================================

    def euler_objective(
        self,
        policy,
        y_t: torch.Tensor,
        w_t: torch.Tensor,
        z_t: torch.Tensor,
        dist_features_t: torch.Tensor,
        w_raw_t: torch.Tensor,
        y_next_1: torch.Tensor,
        w_next_1: torch.Tensor,
        z_next_1: torch.Tensor,
        dist_features_next_1: torch.Tensor,
        w_raw_next_1: torch.Tensor,
        y_next_2: torch.Tensor,
        w_next_2: torch.Tensor,
        z_next_2: torch.Tensor,
        dist_features_next_2: torch.Tensor,
        w_raw_next_2: torch.Tensor,
        R_next_1: float,
        R_next_2: float,
        nu_h: float = 1.0,
        nu: float = 1.0,
        input_scale_spec: InputScaleSpec = InputScaleSpec()
    ) -> torch.Tensor:
        """
        Euler equation objective with heterogeneous agents and AiO operator (Maliar et al., Eq. 44).
        
        Implements the All-in-One (AiO) operator approach: uses two independent shocks
        to efficiently compute nested expectations without explicit integration. This reduces
        the curse of dimensionality in the heterogeneous-agent setting.
        
        Loss function: Ξ(θ) = E[Ψ^FB(a,λ)²] + ν·E[Euler_1·Euler_2]
        
        Where:
        - Ψ^FB: Fischer-Burmeister complementarity function checking KKT conditions
        - Euler_1, Euler_2: Residuals from intertemporal FOC under two shocks
        - a: Saving share = 1 - c/w (constraint: a ≥ 0, i.e., w ≥ 0)
        - λ: Complementarity multiplier = 1 - h (constraint: λ ≥ 0)
        - h: Lagrange multiplier from policy network output
        
        Args:
            policy: KSNeuralNetworkPolicy for computing (c, h).
            y_t, w_t, z_t, dist_features_t, w_raw_t: Current state at time t.
                - y_t: Idiosyncratic productivity, shape (batch_size, num_agents)
                - w_t: Scaled wealth, shape (batch_size, num_agents)
                - z_t: Aggregate log-TFP, shape (batch_size,)
                - dist_features_t: Distribution features D_t
                - w_raw_t: Unscaled wealth for complementarity check
            y_next_1, ..., w_raw_next_1: State under first shock at t+1.
            y_next_2, ..., w_raw_next_2: State under second shock at t+1.
            R_next_1: Gross interest rate for shock 1 scenario.
            R_next_2: Gross interest rate for shock 2 scenario.
            nu_h: Deprecated parameter (for API compatibility).
            nu: Weight on Euler loss relative to Fischer-Burmeister loss. Default: 1.0.
            input_scale_spec: InputScaleSpec for wealth scaling (e.g., log-scale bounds).
        
        Returns:
            torch.Tensor: Scalar loss to minimize.
        """
        eps_guard = 1e-10
        w_cap = float(input_scale_spec.w_max) if input_scale_spec.enabled else None
        
        # ===== Current state =====
        c_t = consumption_from_share_torch(
            policy, y_t, w_t, z_t, dist_features_t, w_raw_t, w_cap=w_cap
        )
        h_t = policy.forward_h(y_t, w_t, z_t, dist_features_t)
        u_c_t = c_t**(-self.model.params.gamma)
        
        # ===== Next period shock 1 =====
        c_next_1 = consumption_from_share_torch(
            policy, y_next_1, w_next_1, z_next_1, dist_features_next_1,
            w_raw_next_1, w_cap=w_cap
        )
        u_c_next_1 = c_next_1**(-self.model.params.gamma)
        
        # ===== Next period shock 2 =====
        c_next_2 = consumption_from_share_torch(
            policy, y_next_2, w_next_2, z_next_2, dist_features_next_2,
            w_raw_next_2, w_cap=w_cap
        )
        u_c_next_2 = c_next_2**(-self.model.params.gamma)
        
        # ===== Fischer-Burmeister complementarity: Ψ^FB(a, λ) (Paper Eq. 44) =====
        # a = 1 - c/w (saving share)
        # λ = 1 - h (complementarity multiplier)
        eps_guard_div = torch.clamp(w_raw_t, min=eps_guard)
        a = torch.clamp(1.0 - c_t / eps_guard_div, min=0.0)
        lambda_val = torch.clamp(1.0 - h_t, min=0.0)
        
        # FB residual (squared as per Paper Eq. 44)
        fb_residual = self._fischer_burmeister_batch(a, lambda_val)
        
        # ===== Euler equation residuals for both shocks (Paper Eq. 44) =====
        # These should be zero if h is correct and FOCs are satisfied
        euler_1 = (self.model.params.beta * R_next_1 * u_c_next_1 / 
                   (u_c_t + eps_guard) - h_t)
        euler_2 = (self.model.params.beta * R_next_2 * u_c_next_2 / 
                   (u_c_t + eps_guard) - h_t)
        
        # ===== Loss (Paper Eq. 44): FB term squared, Euler term as AiO product =====
        loss_fb = torch.mean(fb_residual**2)
        loss_euler = torch.mean(euler_1 * euler_2)
        
        total_loss = loss_fb + nu * loss_euler
        
        return total_loss

    # =====================================================================
    # Bellman Equation Objective
    # =====================================================================

    def bellman_objective(
        self,
        policy,
        y_t: torch.Tensor,
        w_t: torch.Tensor,
        z_t: torch.Tensor,
        dist_features_t: torch.Tensor,
        w_raw_t: torch.Tensor,
        y_next_1: torch.Tensor,
        w_next_1: torch.Tensor,
        z_next_1: torch.Tensor,
        dist_features_next_1: torch.Tensor,
        w_raw_next_1: torch.Tensor,
        y_next_2: torch.Tensor,
        w_next_2: torch.Tensor,
        z_next_2: torch.Tensor,
        dist_features_next_2: torch.Tensor,
        w_raw_next_2: torch.Tensor,
        R_next_1: float,
        R_next_2: float,
        nu_h: float = 1.0,
        nu: float = 1.0,
        input_scale_spec: InputScaleSpec = InputScaleSpec()
    ) -> torch.Tensor:
        """
        Bellman equation objective with heterogeneous agents and AiO operator (Maliar et al., Eq. 32).
        
        Trains the value function approximation V_θ to match the Bellman equation in expectation.
        Uses the AiO operator with two independent shocks to efficiently compute nested 
        expectations. Also enforces envelope condition and complementarity constraints.
        
        Loss function (AiO structure):
            Ξ(θ) = E[Bellman_1·Bellman_2] + ν·E[FB_1·FB_2] + ν_h·E[(λ_1·a_1)·(λ_2·a_2)]
        
        Where:
        - Bellman residuals: ε_Bellman = V(s) - u(c(s)) - β·E[V(s')]
        - FB and complementarity terms enforce KKT constraints as in Euler objective
        - Envelope condition uses finite differences: ∂V/∂w ≈ ∂u/∂c (at optimum)
        
        Args:
            policy: KSNeuralNetworkPolicy with value function head V_θ.
            y_t, w_t, z_t, dist_features_t, w_raw_t: Current state at time t.
            y_next_1, ..., w_raw_next_1: State under first shock at t+1.
            y_next_2, ..., w_raw_next_2: State under second shock at t+1.
            nu_h: Weight on multiplier consistency term (λ·a product). Default: 1.0.
            nu: Weight on complementarity/envelope terms relative to Bellman. Default: 1.0.
            input_scale_spec: Wealth scaling specification for unscaling and constraint checks.
        
        Returns:
            torch.Tensor: Scalar loss to minimize.
        """
        eps_guard = 1e-10
        w_cap = float(input_scale_spec.w_max) if input_scale_spec.enabled else None
        
        # ===== Current state outputs =====
        c_t = consumption_from_share_torch(
            policy, y_t, w_t, z_t, dist_features_t, w_raw_t, w_cap=w_cap
        )
        h_t = policy.forward_h(y_t, w_t, z_t, dist_features_t)
        V_t = policy.forward_v(y_t, w_t, z_t, dist_features_t)
        
        # Utility at current consumption
        gamma = self.model.params.gamma
        if abs(gamma - 1.0) < 1e-10:
            u_t = torch.log(torch.clamp(c_t, min=1e-8))
            u_c_t = 1.0 / torch.clamp(c_t, min=1e-8)
        else:
            u_t = (c_t**(1 - gamma) - 1) / (1 - gamma)
            u_c_t = c_t**(-gamma)
        
        # ===== Shock 1: First AiO term =====
        V_next_1 = policy.forward_v(
            y_next_1, w_next_1, z_next_1, dist_features_next_1
        )
        c_next_1 = consumption_from_share_torch(
            policy, y_next_1, w_next_1, z_next_1, dist_features_next_1,
            w_raw_next_1, w_cap=w_cap
        )
        
        # ===== Shock 2: Second AiO term (independent) =====
        V_next_2 = policy.forward_v(
            y_next_2, w_next_2, z_next_2, dist_features_next_2
        )
        c_next_2 = consumption_from_share_torch(
            policy, y_next_2, w_next_2, z_next_2, dist_features_next_2,
            w_raw_next_2, w_cap=w_cap
        )
        
        # ===== Bellman residual (AiO product) =====
        bellman_1 = V_t - u_t - self.model.params.beta * V_next_1
        bellman_2 = V_t - u_t - self.model.params.beta * V_next_2
        loss_bellman = torch.mean(bellman_1 * bellman_2)
        
        # ===== Value function gradient using forward differences (Section 4 compatible) =====
        dV_dw_1 = self._value_derivative_w(
            policy,
            y_next_1, w_next_1, z_next_1, dist_features_next_1,
            h_value=1e-4
        )
        dV_dw_2 = self._value_derivative_w(
            policy,
            y_next_2, w_next_2, z_next_2, dist_features_next_2,
            h_value=1e-4
        )
        
        # ===== Complementary slackness: a·λ = 0 with a,λ ≥ 0 =====
        w_safe = torch.clamp(w_raw_t, min=eps_guard)
        a = torch.clamp(1.0 - c_t / w_safe, min=0.0)
        
        # ===== Lagrange multiplier from FOC (Section 4 pattern, with interest rates) =====
        # λ = 1 - (β·R·∂V/∂w')/u'(c)
        # For Krusell-Smith: Interest rates R are computed from equilibrium factor prices
        # R_t = 1 - δ + z_t·α·K_t^(α-1)·L_t^(1-α) (see model.factor_prices)
        # Pass R from caller (computed from aggregate capital and labor in main_section5.py)
        R_t_1 = R_next_1
        R_t_2 = R_next_2
        
        u_c_next_1 = c_next_1**(-gamma)
        u_c_next_2 = c_next_2**(-gamma)
        
        # Corrected FOC: λ = 1 - (β·R·∂V/∂w')/u'(c)
        lambda_1 = 1.0 - (self.model.params.beta * R_t_1 * dV_dw_1) / (u_c_t + eps_guard)
        lambda_2 = 1.0 - (self.model.params.beta * R_t_2 * dV_dw_2) / (u_c_t + eps_guard)
        lambda_1 = torch.clamp(lambda_1, min=0.0)
        lambda_2 = torch.clamp(lambda_2, min=0.0)
        
        # ===== Fischer-Burmeister complementarity (Paper Eq. 45: squared) =====
        fb = self._fischer_burmeister_batch(a, lambda_1)  # Single FB for all shocks
        loss_fb = torch.mean(fb**2)  # Squared as per Paper Eq. 45
        
        # ===== Multiplier consistency across shocks (AiO product, Paper Eq. 45) =====
        # FOC residual: β·∂V/∂w'/u'(c) - h
        mult_1 = (self.model.params.beta * dV_dw_1 / (u_c_t + eps_guard) - h_t)
        mult_2 = (self.model.params.beta * dV_dw_2 / (u_c_t + eps_guard) - h_t)
        loss_mult = torch.mean(mult_1 * mult_2)  # AiO product
        
        # ===== Combined loss (Paper Eq. 45) =====
        total_loss = loss_bellman + nu * loss_fb + nu_h * loss_mult
        
        return total_loss

    def _fischer_burmeister_batch(
        self,
        a: torch.Tensor,
        lambda_val: torch.Tensor
    ) -> torch.Tensor:
        """
        Fischer-Burmeister complementarity function for constraint violation detection.
        
        Mathematical form: Ψ^FB(a,λ) = a + λ - √(a² + λ²)
        
        Properties:
        - Ψ^FB(a,λ) = 0 if and only if a·λ = 0 with a,λ ≥ 0 (complementarity)
        - Used in Euler and Bellman objectives to enforce Kuhn-Tucker conditions
        - Reference: Paper Eq. 44-45, Maliar et al. (2021)
        
        Args:
            a: Saving share constraint variable (non-negative), shape (batch_size, ...).
            lambda_val: Complementarity multiplier (non-negative), same shape.
        
        Returns:
            torch.Tensor: FB function values, same shape as inputs.
        """
        return a + lambda_val - torch.sqrt(a ** 2 + lambda_val ** 2 + 1e-12)

    def _value_derivative_w(
        self,
        policy,
        y: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        dist_features: torch.Tensor,
        h_value: float = 1e-4,
        w_range: float = 1.0
    ) -> torch.Tensor:
        """
        Approximate ∂V/∂w using central differences for better numerical stability.
        
        Central difference formula: ∂V/∂w ≈ [V(y, w+h) - V(y, w-h)] / (2h)
        
        This is more accurate and stable than forward difference for value functions
        with significant curvature. Critical for Bellman objective where the envelope
        theorem requires ∂V/∂w = u'(c) at the optimum.
        
        Args:
            policy: KSNeuralNetworkPolicy with value function head.
            y, w, z, dist_features: Current state at which to compute gradient.
                - y: Productivity, shape (batch_size, num_agents)
                - w: Wealth (could be scaled), shape (batch_size, num_agents)
                - z: Aggregate log-TFP, shape (batch_size,)
                - dist_features: Distribution features D_t
            h_value: Step size for finite difference. Default: 1e-4.
                Adaptive scaling: multiplied by w_range for scale-dependent grids.
            w_range: Scaling factor for step size (typically 1.0 or mean wealth).
        
        Returns:
            torch.Tensor: Approximate gradient ∂V/∂w, shape matching (batch_size, num_agents).
        """
        # Ensure w is positive and clamp to valid range
        w_safe = torch.clamp(w, min=1e-8)
        
        # Adaptive step size based on wealth level
        # Use smaller relative step for larger wealth to maintain precision
        h_abs = h_value * torch.maximum(torch.ones_like(w_safe), w_safe / 10.0)
        
        w_plus = w_safe + h_abs
        w_minus = torch.clamp(w_safe - h_abs, min=1e-8)

        V_plus = policy.forward_v(y, w_plus, z, dist_features)
        V_minus = policy.forward_v(y, w_minus, z, dist_features)

        # Central difference
        dV_dw = (V_plus - V_minus) / (2.0 * h_abs)
        
        # Guard against non-finite gradients
        dV_dw = torch.where(
            torch.isfinite(dV_dw),
            dV_dw,
            torch.zeros_like(dV_dw)
        )
        
        # Clamp gradients to reasonable range to prevent extreme multiplier values
        dV_dw = torch.clamp(dV_dw, min=-10.0, max=10.0)
        
        return dV_dw

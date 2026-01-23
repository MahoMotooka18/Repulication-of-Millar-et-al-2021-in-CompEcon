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
    """Computes loss objectives for KS training."""

    def __init__(
        self,
        model: KrusellSmithModel,
        device: str = 'cpu'
    ):
        """
        Initialize objective computer.
        
        Args:
            model: KrusellSmithModel instance
            device: 'cpu' or 'cuda'
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
        Lifetime Reward objective.
        
        Maximizes sum_t beta^t u(c_t^i) for each agent i.
        
        Args:
            c_path: consumption (T, batch_size, num_agents)
            w_path: wealth (T, batch_size, num_agents)
            
        Returns:
            scalar loss (negative of reward)
        """
        T = c_path.shape[0]
        gamma = self.model.params.gamma
        
        # Compute utilities
        if abs(gamma - 1.0) < 1e-10:
            utilities = torch.log(c_path + 1e-8)
        else:
            utilities = (c_path**(1 - gamma) - 1) / (1 - gamma)
        
        # Discount and sum
        discount_factors = torch.tensor(
            [self.model.params.beta**t for t in range(T)],
            device=self.device,
            dtype=torch.float32
        ).view(-1, 1, 1)
        
        lifetime_reward = torch.sum(discount_factors * utilities, dim=0)
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
        Euler equation objective with two uncorrelated shocks (Paper Eq. 44).
        
        Ξ(θ) = E { [Ψ^FB(...)]² + ν·[Euler_1]·[Euler_2] }
        
        Args:
            policy: KSNeuralNetworkPolicy
            Current state: (y_t, w_t, z_t, dist_features_t, w_raw_t)
            Next period: shock 1 and shock 2 with corresponding interest rates
            nu_h: (not used in Euler, for API consistency)
            nu: weight on Euler term relative to FB term
            
        Returns:
            scalar loss to minimize
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
    # Bellman Equation Objective (Corrected)
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
        nu_h: float = 1.0,
        nu: float = 1.0,
        input_scale_spec: InputScaleSpec = InputScaleSpec()
    ) -> torch.Tensor:
        """
        Bellman equation objective with two uncorrelated shocks (AiO).
        
        Implements AiO structure matching Section 4 (Consumption-Saving):
        Ξ(θ) = E { [Bellman_1 × Bellman_2]
                   + ν·[FB_1 × FB_2]
                   + ν_h·[(λ_1 × a) × (λ_2 × a)] }
        
        Args:
            policy: KSNeuralNetworkPolicy
            Current and next-period state variables for two independent shocks
            nu: weight on Fischer-Burmeister complementarity term
            nu_h: weight on multiplier consistency term
            
        Returns:
            scalar loss to minimize
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
        
        # ===== Lagrange multiplier from FOC (Section 4 pattern) =====
        # λ = 1 - (β·R·∂V/∂w')/u'(c)
        # For Krusell-Smith: R varies, so use interest rate from state
        # Conservative: use average or extract from rate calculation
        u_c_next_1 = c_next_1**(-gamma)
        u_c_next_2 = c_next_2**(-gamma)
        
        lambda_1 = 1.0 - (self.model.params.beta * dV_dw_1) / (u_c_t + eps_guard)
        lambda_2 = 1.0 - (self.model.params.beta * dV_dw_2) / (u_c_t + eps_guard)
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
        Fischer-Burmeister complementarity function (Section4-consistent).
        
        Ψ^FB(a,λ) = a + λ - √(a² + λ²)
        
        Ensures a·λ = 0 (complementarity) with a,λ ≥ 0 (non-negativity).
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
        Approximate dV/dw using forward differences (Section 4 compatible).
        
        Uses forward difference to avoid negative w values:
        dV/dw ≈ (V(y, w+h) - V(y, w)) / h
        
        This ensures w remains in the valid domain [0, ∞).
        """
        # Ensure w is positive
        w_safe = torch.clamp(w, min=1e-10)
        w_plus = w_safe + h_value

        V_base = policy.forward_v(y, w_safe, z, dist_features)
        V_plus = policy.forward_v(y, w_plus, z, dist_features)

        dV_dw = (V_plus - V_base) / h_value
        
        # Guard against non-finite gradients
        dV_dw = torch.where(
            torch.isfinite(dV_dw),
            dV_dw,
            torch.zeros_like(dV_dw)
        )
        
        return dV_dw

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
        y_next_1: torch.Tensor,
        w_next_1: torch.Tensor,
        z_next_1: torch.Tensor,
        dist_features_next_1: torch.Tensor,
        y_next_2: torch.Tensor,
        w_next_2: torch.Tensor,
        z_next_2: torch.Tensor,
        dist_features_next_2: torch.Tensor,
        R_next_1: float,
        R_next_2: float,
        nu_h: float = 1.0
    ) -> torch.Tensor:
        """
        Euler equation objective with two uncorrelated shocks.
        
        Args:
            policy: KSNeuralNetworkPolicy
            y_t: productivity (batch_size,)
            w_t: wealth (batch_size,)
            z_t: aggregate productivity (batch_size,)
            dist_features_t: distribution features (batch_size, dist_feat)
            eps1_y: idiosyncratic shock 1
            eps2_y: idiosyncratic shock 2
            eps_z: aggregate shock
            R_next: interest rate for next period
            W_next: wage for next period
            nu_h: weight on multiplier term
            
        Returns:
            scalar loss
        """
        # Current consumption and multiplier
        c_t = policy.forward_policy(y_t, w_t, z_t, dist_features_t)
        h_t = policy.forward_h(y_t, w_t, z_t, dist_features_t)
        
        # Marginal utility at c_t
        u_c_t = c_t**(-self.model.params.gamma)
        
        c_next_1 = policy.forward_policy(
            y_next_1, w_next_1, z_next_1, dist_features_next_1
        )
        u_c_next_1 = c_next_1**(-self.model.params.gamma)
        
        c_next_2 = policy.forward_policy(
            y_next_2, w_next_2, z_next_2, dist_features_next_2
        )
        u_c_next_2 = c_next_2**(-self.model.params.gamma)
        
        # Fischer-Burmeister residual
        a = 1.0 - c_t / w_t
        fb_residual = a + (1.0 - h_t) - torch.sqrt(
            a**2 + (1.0 - h_t)**2 + 1e-12
        )
        
        # Euler expectation terms
        euler_1 = (self.model.params.beta * R_next_1 * u_c_next_1 /
                   u_c_t - h_t)
        euler_2 = (self.model.params.beta * R_next_2 * u_c_next_2 /
                   u_c_t - h_t)
        
        loss_fb = torch.mean(fb_residual**2)
        loss_euler = torch.mean(euler_1 * euler_2)
        
        total_loss = loss_fb + nu_h * loss_euler
        
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
        y_next_1: torch.Tensor,
        w_next_1: torch.Tensor,
        z_next_1: torch.Tensor,
        dist_features_next_1: torch.Tensor,
        y_next_2: torch.Tensor,
        w_next_2: torch.Tensor,
        z_next_2: torch.Tensor,
        dist_features_next_2: torch.Tensor,
        nu_h: float = 1.0,
        nu: float = 1.0
    ) -> torch.Tensor:
        """
        Bellman equation objective with two uncorrelated shocks.
        
        Args:
            policy: KSNeuralNetworkPolicy (must have all outputs)
            (other args same as euler_objective)
            nu: weight on FB term
            
        Returns:
            scalar loss
        """
        # Current state outputs
        c_t, phi_t, h_t, V_t = policy.forward_all(
            y_t, w_t, z_t, dist_features_t
        )
        
        # Utility at current consumption
        gamma = self.model.params.gamma
        if abs(gamma - 1.0) < 1e-10:
            u_c_t = 1.0 / c_t
            u_t = torch.log(c_t + 1e-8)
        else:
            u_c_t = c_t**(-gamma)
            u_t = (c_t**(1 - gamma) - 1) / (1 - gamma)
        
        # Next period with shock 1
        V_next_1 = policy.forward_v(
            y_next_1, w_next_1, z_next_1, dist_features_next_1
        )
        c_next_1 = policy.forward_policy(
            y_next_1, w_next_1, z_next_1, dist_features_next_1
        )
        u_c_next_1 = c_next_1**(-gamma)
        
        # Next period with shock 2
        V_next_2 = policy.forward_v(
            y_next_2, w_next_2, z_next_2, dist_features_next_2
        )
        c_next_2 = policy.forward_policy(
            y_next_2, w_next_2, z_next_2, dist_features_next_2
        )
        u_c_next_2 = c_next_2**(-gamma)
        
        # Bellman residual
        bellman_1 = V_t - u_t - self.model.params.beta * V_next_1
        bellman_2 = V_t - u_t - self.model.params.beta * V_next_2
        
        loss_bellman = torch.mean(bellman_1 * bellman_2)
        
        # FB residual
        a = 1.0 - c_t / w_t
        fb_residual = a + (1.0 - h_t) - torch.sqrt(
            a**2 + (1.0 - h_t)**2 + 1e-12
        )
        
        loss_fb = torch.mean(fb_residual**2)
        
        # Value derivative matching (finite differences)
        dV_dw_1 = self._value_derivative_w(
            policy, y_next_1, w_next_1, z_next_1, dist_features_next_1
        )
        dV_dw_2 = self._value_derivative_w(
            policy, y_next_2, w_next_2, z_next_2, dist_features_next_2
        )
        
        mult_1 = (self.model.params.beta * dV_dw_1 / u_c_t - h_t)
        mult_2 = (self.model.params.beta * dV_dw_2 / u_c_t - h_t)
        
        loss_mult = torch.mean(mult_1 * mult_2)
        
        total_loss = loss_bellman + nu * loss_fb + nu_h * loss_mult
        
        return total_loss

    def _value_derivative_w(
        self,
        policy,
        y: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        dist_features: torch.Tensor,
        h_value: float = 1e-4
    ) -> torch.Tensor:
        """Approximate dV/dw using finite differences."""
        w_plus = w + h_value
        w_minus = w - h_value
        
        V_plus = policy.forward_v(y, w_plus, z, dist_features)
        V_minus = policy.forward_v(y, w_minus, z, dist_features)
        
        return (V_plus - V_minus) / (2 * h_value)

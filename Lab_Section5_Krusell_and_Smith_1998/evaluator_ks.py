"""
Evaluator for Krusell-Smith Model

Computes evaluation metrics and summary statistics.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from model_ks1998 import KrusellSmithModel
from policy_utils_ks import (
    PolicyOutputType,
    InputScaleSpec,
    scale_inputs_numpy,
    consumption_from_share_torch,
    build_dist_features_numpy
)


class KSEvaluator:
    """Evaluation and metrics computation for KS model."""

    def __init__(
        self,
        model: KrusellSmithModel,
        device: str = 'cpu',
        input_scale_spec: InputScaleSpec = InputScaleSpec(),
        policy_output_type: str = PolicyOutputType.C_SHARE
    ):
        """
        Initialize evaluator.
        
        Args:
            model: KrusellSmithModel instance
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.input_scale_spec = input_scale_spec
        self.policy_output_type = policy_output_type

    def evaluate_simulation(
        self,
        policy,
        w_init: np.ndarray,
        y_init: np.ndarray,
        z_init: float,
        K_init: float,
        T: int = 1000,
        seed: int = 0
    ) -> Dict:
        """
        Evaluate policy through simulation.
        
        Args:
            policy: neural network policy
            w_init: initial wealth distribution
            y_init: initial productivity distribution
            z_init: initial aggregate productivity
            K_init: initial aggregate capital
            T: simulation length
            seed: random seed
            
        Returns:
            dict with simulation results
        """
        rng = np.random.default_rng(seed)
        num_agents = len(w_init)
        
        # Initialize
        w_t = w_init.copy()
        y_t = y_init.copy()
        z_t = z_init
        K_t = K_init
        
        # Storage
        w_path = np.zeros((T, num_agents))
        y_path = np.zeros((T, num_agents))
        c_path = np.zeros((T, num_agents))
        k_path = np.zeros((T, num_agents))
        K_path = np.zeros(T)
        z_path = np.zeros(T)
        Y_path = np.zeros(T)
        C_path = np.zeros(T)
        R_path = np.zeros(T)
        W_path = np.zeros(T)
        
        gamma = self.model.params.gamma
        
        with torch.no_grad():
            for t in range(T):
                # Normalize productivity
                y_t = self.model.normalize_productivity(y_t)
                
                # Scaled inputs for policy
                y_scaled, w_scaled, z_scaled = scale_inputs_numpy(
                    y_t, w_t, z_t, self.input_scale_spec
                )

                # Distribution vector (scaled)
                dist_vec = build_dist_features_numpy(y_scaled, w_scaled)
                
                # Convert to tensors
                y_tensor = torch.from_numpy(y_scaled).float().to(self.device)
                w_tensor = torch.from_numpy(w_scaled).float().to(self.device)
                z_tensor = torch.full((num_agents,), z_scaled,
                                     dtype=torch.float32,
                                     device=self.device)
                w_raw_tensor = torch.from_numpy(w_t).float().to(self.device)
                dist_tensor = torch.from_numpy(
                    dist_vec
                ).float().to(self.device).unsqueeze(0).expand(
                    num_agents, -1
                )
                
                # Get consumption
                if self.policy_output_type != PolicyOutputType.C_SHARE:
                    raise ValueError(
                        "Only c_share policy output is supported with input scaling."
                    )
                c_t = consumption_from_share_torch(
                    policy, y_tensor, w_tensor, z_tensor, dist_tensor, w_raw_tensor
                )
                c_t = torch.clamp(
                    c_t,
                    min=torch.zeros_like(w_raw_tensor),
                    max=w_raw_tensor
                )
                c_t_np = c_t.cpu().numpy()
                k_next = w_t - c_t_np
                
                # Store
                w_path[t] = w_t
                y_path[t] = y_t
                c_path[t] = c_t_np
                k_path[t] = k_next
                K_path[t] = K_t
                z_path[t] = z_t
                
                # Transition
                eps_y = rng.standard_normal(num_agents)
                eps_z = rng.standard_normal()
                
                y_next = self.model.income_transition(y_t, eps_y)
                z_next = self.model.aggregate_productivity_transition(
                    z_t, eps_z
                )
                y_next = self.model.normalize_productivity(y_next)
                K_next = np.sum(k_next)
                L_next = self.model.total_labor(y_next)
                R_next, W_next = self.model.factor_prices(
                    z_next, K_next, L_next
                )
                Y_path[t] = self.model.production_output(
                    z_t, K_t, self.model.total_labor(y_t)
                )
                C_path[t] = np.sum(c_t_np)
                w_next = self.model.state_transition(
                    w_t, c_t_np, y_next, R_next, W_next
                )
                R_path[t] = R_next
                W_path[t] = W_next
                
                w_t = w_next
                y_t = y_next
                z_t = z_next
                K_t = K_next
        
        return {
            'w_path': w_path,
            'y_path': y_path,
            'c_path': c_path,
            'k_path': k_path,
            'K_path': K_path,
            'z_path': z_path,
            'Y_path': Y_path,
            'C_path': C_path,
            'R_path': R_path,
            'W_path': W_path
        }

    def compute_statistics(
        self,
        simulation: Dict,
        burn_in: int = 100
    ) -> Dict[str, float]:
        """
        Compute summary statistics from simulation.
        
        Args:
            simulation: output from evaluate_simulation
            burn_in: number of periods to discard
            
        Returns:
            dict with statistics
        """
        w_path = simulation['w_path'][burn_in:]
        y_path = simulation['y_path'][burn_in:]
        c_path = simulation['c_path'][burn_in:]
        k_path = simulation['k_path'][burn_in:]
        K_path = simulation['K_path'][burn_in:]
        z_path = simulation['z_path'][burn_in:]
        Y_path = simulation['Y_path'][burn_in:]
        C_path = simulation['C_path'][burn_in:]
        R_path = simulation['R_path'][burn_in:]
        
        # Aggregate statistics
        y_std = float(np.std(Y_path))
        corr_yc = float(np.corrcoef(Y_path, C_path)[0, 1])
        
        # Wealth inequality
        w_flat = k_path.flatten()
        gini = 2 * np.sum((np.arange(1, len(w_flat) + 1)) *
                          np.sort(w_flat)) / (len(w_flat) *
                          np.sum(w_flat)) - (len(w_flat) + 1) / len(w_flat)
        
        # Wealth shares
        total_wealth = np.sum(w_flat)
        w_sorted = np.sort(w_flat)
        n = len(w_flat)
        
        share_bottom_40 = np.sum(w_sorted[:int(0.4 * n)]) / total_wealth
        share_top_20 = np.sum(w_sorted[-int(0.2 * n):]) / total_wealth
        share_top_1 = np.sum(w_sorted[-int(0.01 * n):]) / total_wealth
        
        # KS regression: ln(k_{t+1}) = xi_0 + xi_1 ln(k_t) + xi_2 ln(z_t)
        k_path = K_path
        k_t = np.log(k_path[:-1] + 1e-6)
        k_next = np.log(k_path[1:] + 1e-6)
        z_t = z_path[:-1]
        
        # Regression
        X = np.column_stack([np.ones(len(k_t)), k_t, z_t])
        y_reg = k_next
        
        try:
            beta_hat = np.linalg.lstsq(X, y_reg, rcond=None)[0]
            y_pred = X @ beta_hat
            ss_res = np.sum((y_reg - y_pred)**2)
            ss_tot = np.sum((y_reg - np.mean(y_reg))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        except:
            r2 = 0.0
        
        return {
            'std_y': y_std,
            'corr_y_c': corr_yc,
            'gini_k': float(gini),
            'share_bottom_40': float(share_bottom_40),
            'share_top_20': float(share_top_20),
            'share_top_1': float(share_top_1),
            'K_mean': float(np.mean(K_path)),
            'K_std': float(np.std(K_path)),
            'r2': float(r2)
        }

    def compute_euler_residuals(
        self,
        simulation: Dict,
        burn_in: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute Euler equation residuals.
        
        Args:
            simulation: output from evaluate_simulation
            burn_in: number of periods to discard
            
        Returns:
            dict with residuals
        """
        w_path = simulation['w_path'][burn_in:-1]
        c_path = simulation['c_path'][burn_in:-1]
        c_next_path = simulation['c_path'][burn_in+1:]
        R_next_path = simulation['R_path'][burn_in:-1]
        
        gamma = self.model.params.gamma
        
        # Marginal utilities
        u_c = c_path**(-gamma)
        u_c_next = c_next_path**(-gamma)
        
        # Euler residual
        euler_residual = u_c - (self.model.params.beta *
                                R_next_path[:, None] * u_c_next)
        
        euler_residual_abs = np.abs(euler_residual)
        
        return {
            'euler_residual': euler_residual,
            'euler_residual_abs': euler_residual_abs,
            'euler_residual_mean': float(np.mean(euler_residual_abs)),
            'euler_residual_p50': float(np.percentile(
                euler_residual_abs, 50
            )),
            'euler_residual_p90': float(np.percentile(
                euler_residual_abs, 90
            ))
        }

    def compute_lifetime_reward(
        self,
        simulation: Dict,
        burn_in: int = 100
    ) -> Dict[str, float]:
        """
        Compute discounted lifetime reward from simulated consumption.
        """
        c_path = simulation['c_path'][burn_in:]
        T = c_path.shape[0]
        gamma = self.model.params.gamma

        if abs(gamma - 1.0) < 1e-10:
            u_path = np.log(np.maximum(c_path, 1e-12))
        else:
            u_path = (c_path**(1 - gamma) - 1) / (1 - gamma)

        discounts = np.array([self.model.params.beta**t for t in range(T)])
        rewards = np.sum(u_path * discounts[:, None, None], axis=0)
        return {
            'lifetime_reward_mean': float(np.mean(rewards)),
            'lifetime_reward_std': float(np.std(rewards))
        }

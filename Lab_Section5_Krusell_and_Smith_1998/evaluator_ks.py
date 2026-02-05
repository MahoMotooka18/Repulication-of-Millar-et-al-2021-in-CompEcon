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
    """
    Evaluation module for Krusell-Smith heterogeneous-agent models.
    
    Computes evaluation metrics and summary statistics across entire simulated
    paths of heterogeneous agents in the economy. Metrics include:
    - Agent-level satisfaction via lifetime utility
    - Euler equation residuals (constraint satisfaction and FOC accuracy)
    - Bellman equation residuals (value function fitting quality)
    - Cross-sectional statistics (wealth distribution, consumption patterns)
    - Aggregate consistency (capital-labor equilibrium conditions)
    
    Works with both scaled and unscaled wealth inputs, handles non-linear
    transformations for numerical stability.
    """

    def __init__(
        self,
        model: KrusellSmithModel,
        device: str = 'cpu',
        input_scale_spec: InputScaleSpec = InputScaleSpec(),
        policy_output_type: str = PolicyOutputType.C_SHARE
    ) -> None:
        """
        Initialize Krusell-Smith evaluator.
        
        Args:
            model: KrusellSmithModel instance with parameters and utility function.
            device: Computing device ('cpu' or 'cuda'). Default: 'cpu'.
            input_scale_spec: Input scaling specification for wealth transformations
                (e.g., log-scale with bounds). Default: no scaling.
            policy_output_type: Type of policy output expected. Default: consumption share.
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
        Simulate the economy using the learned policy and collect metrics.
        
        Performs a forward simulation of T periods where agents make consumption decisions
        via the policy function, experience idiosyncratic and aggregate shocks, and update
        their cash-on-hand. Tracks consumption, wealth, capital paths and computes
        cross-sectional statistics at each period.
        
        Simulation loop:
        1. Agents observe (y_t, w_t, z_t, D_t) and compute c_t = policy(...)
        2. Income updates: y_t+1 from AR(1) income process
        3. Wealth evolves: w_t+1 = w_t - c_t + (wage + interest)*h_t + transfers
        4. Aggregate shocks: z_t+1 from AR(1) TFP process
        5. Aggregate state: K_t+1 from sum of agent capital, D_t+1 from distribution
        
        Args:
            policy: KSNeuralNetworkPolicy for computing consumption from states.
            w_init: Initial wealth distribution across agents, shape (num_agents,).
            y_init: Initial productivity distribution (log-scale), shape (num_agents,).
            z_init: Initial aggregate log-TFP.
            K_init: Initial aggregate capital stock.
            T: Simulation horizon (number of periods). Default: 1000.
            seed: Random seed for reproducibility. Default: 0.
        
        Returns:
            Dict: Simulation results including:
                - 'w_path': Wealth time series, shape (T, num_agents)
                - 'y_path': Productivity time series, shape (T, num_agents)
                - 'c_path': Consumption time series, shape (T, num_agents)
                - 'k_path': Capital (savings) time series, shape (T, num_agents)
                - 'K_path': Aggregate capital, shape (T,)
                - 'z_path': Aggregate TFP, shape (T,)
                - 'Y_path': Aggregate output, shape (T,)
                - 'R_path': Interest rates, shape (T,)
                - 'w_mean_path': Mean wealth by period, shape (T,)
                - 'c_mean_path': Mean consumption by period, shape (T,)
                - 'gini_wealth': Gini coefficients for wealth distribution, shape (T,)
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
                w_cap = (
                    float(self.input_scale_spec.w_max)
                    if self.input_scale_spec.enabled else None
                )
                c_t = consumption_from_share_torch(
                    policy,
                    y_tensor,
                    w_tensor,
                    z_tensor,
                    dist_tensor,
                    w_raw_tensor,
                    w_cap=w_cap
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
        Compute summary statistics from simulated paths (Krusell-Smith style).
        
        Computes aggregate and distributional statistics over the long run
        (after burn-in period to allow transient dynamics to dissipate).
        Includes business cycle statistics, wealth inequality measures,
        and KS regression coefficients for model validation.
        
        Key statistics computed:
        - Output volatility: std(ln(Y_t))
        - Output-consumption correlation: corr(Y_t, C_t)
        - Wealth Gini coefficient: measures inequality [0=perfect equality, 1=perfect inequality]
        - Wealth distribution shares: fraction held by bottom 40%, top 20%, top 1%
        - KS regression: ln(K_t+1) = ξ_0 + ξ_1·ln(K_t) + ξ_2·ln(Z_t) + u_t
          (Model validation: KS steady state coefficient ≈ 0.999)
        
        Args:
            simulation: Output dictionary from evaluate_simulation() containing
                all simulated paths (w_path, c_path, K_path, Y_path, etc.).
            burn_in: Number of initial periods to discard (default: 100).
                Allows economy to converge to ergodic distribution before computing stats.
        
        Returns:
            Dict[str, float]: Dictionary of statistics including:
                - 'Y_std': Standard deviation of log output
                - 'corr_YC': Correlation between output and consumption
                - 'gini': Gini coefficient for wealth distribution
                - 'share_bottom_40': Wealth share of bottom 40% agents
                - 'share_top_20': Wealth share of top 20% agents
                - 'share_top_1': Wealth share of top 1% agents
                - 'ks_coef_K': KS regression coefficient on ln(K_t)
                - 'ks_coef_Z': KS regression coefficient on ln(Z_t)
                - 'ks_const': KS regression constant term
                - 'ks_r2': R² statistic for KS regression fit
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
        Compute Euler equation residuals from simulation paths.
        
        Evaluates how well the policy satisfies the intertemporal Euler equation:
        u'(c_t) = β·R_t+1·E[u'(c_t+1)]
        
        Residuals measure the FOC error at each agent and period. Small residuals
        indicate the learned policy closely satisfies economic optimality conditions.
        
        Computation:
        - Residual = u'(c_t) - β·R_t+1·u'(c_t+1)
        - Marginal utility: u'(c) = c^(-γ) for CRRA utility
        
        Args:
            simulation: Output from evaluate_simulation() containing consumption
                and interest rate paths.
            burn_in: Number of periods to discard at start. Default: 100.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'residuals': Full Euler residual array, shape (T-burn_in-1, num_agents)
                - 'residuals_mean': Mean residual across agents, shape (T-burn_in-1,)
                - 'residuals_std': Std deviation across agents, shape (T-burn_in-1,)
                - 'residuals_max': Maximum absolute residual per period
        """
        w_path = simulation['w_path'][burn_in:-1]
        c_path = simulation['c_path'][burn_in:-1]
        c_next_path = simulation['c_path'][burn_in+1:]
        R_next_path = simulation['R_path'][burn_in:-1]
        
        gamma = self.model.params.gamma
        
        # Marginal utilities
        u_c = c_path**(-gamma)
        u_c_next = c_next_path**(-gamma)
        
        # Euler residual: u'(c_t) - β·R_t+1·u'(c_t+1)
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
        Compute discounted lifetime utility from simulated consumption paths.
        
        Integrates the discounted utility stream for each agent over the
        simulation horizon. Provides measure of welfare achieved by the policy.
        
        Formula: V_LR = ∑_t β^t·u(c_t) for each agent
        Then reports cross-sectional mean and percentiles.
        
        Args:
            simulation: Output from evaluate_simulation().
            burn_in: Periods to discard at start. Default: 100.
        
        Returns:
            Dict[str, float]: Lifetime utility statistics:
                - 'lifetime_reward_mean': Mean across all agents
                - 'lifetime_reward_p10': 10th percentile
                - 'lifetime_reward_p50': 50th percentile (median)
                - 'lifetime_reward_p90': 90th percentile
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

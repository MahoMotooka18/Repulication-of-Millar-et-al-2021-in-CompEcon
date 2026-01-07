"""
Krusell-Smith (1998) Heterogeneous-Agent Model

Implements the heterogeneous-agent economy with aggregate shocks,
following Krusell and Smith (1998).

State variables:
- Individual: idiosyncratic productivity (y^i), cash-on-hand (w^i)
- Aggregate: aggregate productivity (z), capital (K)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict


@dataclass
class KrusellSmithParams:
    """Parameters for Krusell-Smith 1998 model."""
    
    # Preferences
    gamma: float = 1.0  # Risk aversion
    beta: float = 0.96  # Discount factor
    
    # Production
    alpha: float = 0.36  # Capital share
    delta: float = 0.08  # Depreciation rate
    
    # Idiosyncratic process
    rho_y: float = 0.9  # Persistence
    sigma_y: float = 0.2 * np.sqrt(1.0 - 0.9**2)  # Volatility (adjusted)
    
    # Aggregate process
    rho_z: float = 0.95  # Persistence of log TFP
    sigma_z: float = 0.01  # Volatility of log TFP
    
    # Number of agents
    num_agents: int = 1000
    
    # Simulation
    horizon: int = 100
    
    def __post_init__(self):
        """Validate parameters."""
        assert 0 < self.beta < 1, f"beta must be in (0,1), got {self.beta}"
        assert 0 < self.alpha < 1, f"alpha must be in (0,1), got {self.alpha}"
        assert 0 < self.delta < 1, f"delta must be in (0,1), got {self.delta}"
        assert abs(self.rho_y) < 1, f"|rho_y| must be < 1, got {self.rho_y}"
        assert abs(self.rho_z) < 1, f"|rho_z| must be < 1, got {self.rho_z}"


class KrusellSmithModel:
    """
    Krusell-Smith heterogeneous-agent economy model.
    
    Agents maximize:
        E_0 [sum_t beta^t u(c_t^i)]
    
    subject to:
        w_{t+1}^i = R_{t+1}(w_t^i - c_t^i) + W_{t+1} exp(y_{t+1}^i)
        0 <= c_t^i <= w_t^i
    """
    
    def __init__(self, params: KrusellSmithParams):
        """Initialize the model."""
        self.params = params
    
    def utility(self, c: np.ndarray) -> np.ndarray:
        """
        CRRA utility function.
        
        u(c) = (c^(1-gamma) - 1) / (1 - gamma)
        Special case: u(c) = log(c) when gamma = 1
        """
        gamma = self.params.gamma
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c**(1 - gamma) - 1) / (1 - gamma)
    
    def utility_derivative(self, c: np.ndarray) -> np.ndarray:
        """
        Marginal utility: u'(c) = c^(-gamma)
        """
        return c**(-self.params.gamma)
    
    def income_transition(
        self,
        y_t: np.ndarray,
        eps_t: np.ndarray
    ) -> np.ndarray:
        """
        Idiosyncratic productivity process.
        
        y_{t+1}^i = rho_y * y_t^i + sigma_y * eps_t^i
        
        Args:
            y_t: current log-productivity (num_agents,)
            eps_t: shocks N(0,1) (num_agents,)
            
        Returns:
            y_{t+1}: next productivity (num_agents,)
        """
        return self.params.rho_y * y_t + self.params.sigma_y * eps_t
    
    def aggregate_productivity_transition(
        self,
        z_t: float,
        eps_z_t: float
    ) -> float:
        """
        Aggregate productivity process.
        
        z_{t+1} = rho_z * z_t + sigma_z * eps_z_t
        
        Args:
            z_t: current log TFP
            eps_z_t: aggregate shock N(0,1)
            
        Returns:
            z_{t+1}: next log TFP
        """
        return self.params.rho_z * z_t + self.params.sigma_z * eps_z_t
    
    def production_output(self, z_t: float, K_t: float,
                          total_labor: float) -> float:
        """
        Aggregate production.
        
        Y_t = exp(z_t) * K_t^alpha * L_t^(1-alpha)
        where L_t = sum_i exp(y_t^i) (Eq. 42).
        
        Args:
            z_t: aggregate productivity
            K_t: aggregate capital
            total_labor: sum_i exp(y_t^i)
            
        Returns:
            output Y_t
        """
        z_level = np.exp(z_t)
        return z_level * (K_t**self.params.alpha) * (total_labor**(1 - self.params.alpha))
    
    def factor_prices(
        self,
        z_t: float,
        K_t: float,
        total_labor: float
    ) -> Tuple[float, float]:
        """
        Equilibrium factor prices.
        
        R_t = 1 - delta + exp(z_t) * alpha * K_t^(alpha-1) * L_t^(1-alpha)
        W_t = exp(z_t) * (1-alpha) * K_t^alpha * L_t^(-alpha)
        
        Args:
            z_t: aggregate productivity
            K_t: aggregate capital
            total_labor: sum_i exp(y_t^i)
            
        Returns:
            (R_t, W_t): interest rate and wage
        """
        alpha = self.params.alpha
        delta = self.params.delta
        z_level = np.exp(z_t)
        
        # Interest rate
        if K_t > 0:
            R_t = 1 - delta + z_level * alpha * (K_t**(alpha - 1)) * (total_labor**(1 - alpha))
        else:
            R_t = 1 - delta
        
        # Wage
        if K_t > 0 and total_labor > 0:
            W_t = z_level * (1 - alpha) * (K_t**alpha) / (total_labor**alpha)
        else:
            W_t = z_level * (1 - alpha)
        
        return R_t, W_t
    
    def state_transition(
        self,
        w_t: np.ndarray,
        c_t: np.ndarray,
        y_next: np.ndarray,
        R_next: float,
        W_next: float
    ) -> np.ndarray:
        """
        Cash-on-hand transition.
        
        w_{t+1}^i = R_{t+1}(w_t^i - c_t^i) + W_{t+1} * exp(y_{t+1}^i)
        
        Args:
            w_t: current cash-on-hand (num_agents,)
            c_t: consumption (num_agents,)
            y_next: next productivity (num_agents,)
            R_next: interest rate
            W_next: wage
            
        Returns:
            w_{t+1}: next cash-on-hand (num_agents,)
        """
        savings = w_t - c_t
        capital_income = R_next * savings
        labor_income = W_next * np.exp(y_next)
        return capital_income + labor_income
    
    def aggregate_capital(self, w: np.ndarray, c: np.ndarray) -> float:
        """
        Aggregate capital (savings).
        
        K_{t+1} = sum_i (w_t^i - c_t^i)
        
        Args:
            w: wealth (num_agents,)
            c: consumption (num_agents,)
            
        Returns:
            aggregate capital
        """
        return np.sum(w - c)
    
    def total_labor(self, y: np.ndarray) -> float:
        """
        Aggregate labor supply.
        
        L_t = sum_i exp(y_t^i)
        
        Args:
            y: idiosyncratic productivity (num_agents,)
            
        Returns:
            total labor
        """
        return np.sum(np.exp(y))
    
    def normalize_productivity(self, y: np.ndarray) -> np.ndarray:
        """
        Normalize idiosyncratic productivity so cross-sectional mean = 1.

        This ensures sum_i exp(y_t^i) has expected value 1 in steady state.
        
        Args:
            y: log-productivity (num_agents,)
            
        Returns:
            normalized y
        """
        mean_exp = np.mean(np.exp(y))
        return y - np.log(mean_exp)
    
    def simulate_period(
        self,
        policy_fn,
        w_t: np.ndarray,
        y_t: np.ndarray,
        z_t: float,
        eps_y_t: np.ndarray,
        eps_z_t: float,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """
        Simulate one period of the economy.
        
        Args:
            policy_fn: callable(y, w, z) -> c
            w_t: current wealth
            y_t: current productivity
            z_t: current aggregate state
            eps_y_t: idiosyncratic shocks
            eps_z_t: aggregate shock
            rng: random number generator
            
        Returns:
            (w_{t+1}, y_{t+1}, z_{t+1}, K_{t+1}, prices)
        """
        # Normalize productivity before policy evaluation.
        y_t = self.normalize_productivity(y_t)

        # Consumption decision at time t.
        c_t = policy_fn(y_t, w_t, z_t)
        c_t = np.clip(c_t, 0, w_t)

        k_next = w_t - c_t

        # Draw next-period shocks and update states.
        y_next = self.income_transition(y_t, eps_y_t)
        y_next = self.normalize_productivity(y_next)
        z_next = self.aggregate_productivity_transition(z_t, eps_z_t)

        K_next = np.sum(k_next)
        L_next = self.total_labor(y_next)
        R_next, W_next = self.factor_prices(z_next, K_next, L_next)

        w_next = self.state_transition(w_t, c_t, y_next, R_next, W_next)

        return w_next, y_next, z_next, K_next, (R_next, W_next)
    
    def compute_aggregates(
        self,
        w: np.ndarray,
        y: np.ndarray,
        c: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute aggregate statistics.
        
        Args:
            w: wealth (num_agents,)
            y: productivity (num_agents,)
            c: consumption (num_agents,)
            
        Returns:
            dict with aggregates
        """
        K = self.aggregate_capital(w, c)
        L = self.total_labor(y)
        
        # Gini coefficient
        w_sorted = np.sort(w)
        n = len(w)
        gini = 2 * np.sum((np.arange(1, n + 1)) * w_sorted) / (n * np.sum(w_sorted)) - (n + 1) / n
        
        # Wealth shares
        total_wealth = np.sum(w)
        w_sorted_desc = -np.sort(-w)
        share_bottom_40 = np.sum(w_sorted[:int(0.4 * n)]) / total_wealth if total_wealth > 0 else 0
        share_top_20 = np.sum(w_sorted_desc[:int(0.2 * n)]) / total_wealth if total_wealth > 0 else 0
        share_top_1 = np.sum(w_sorted_desc[:int(0.01 * n)]) / total_wealth if total_wealth > 0 else 0
        
        return {
            'capital': K,
            'labor': L,
            'gini': gini,
            'share_bottom_40': share_bottom_40,
            'share_top_20': share_top_20,
            'share_top_1': share_top_1,
            'wealth_mean': np.mean(w),
            'wealth_std': np.std(w),
            'consumption_mean': np.mean(c),
            'output_per_capita': K**(self.params.alpha) * L**(1 - self.params.alpha)
        }

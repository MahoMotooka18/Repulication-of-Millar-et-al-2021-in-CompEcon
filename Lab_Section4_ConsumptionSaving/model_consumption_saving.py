"""
Consumption-Saving Problem Model

Implements the consumption-saving problem with borrowing constraint as specified in
Section 4 of "Deep learning for solving dynamic economic models" (Maliar et al., 2021).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ConsumptionSavingParams:
    """Parameters for the consumption-saving problem."""
    gamma: float = 2.0  # Risk aversion (CRRA)
    beta: float = 0.9  # Discount factor
    r: float = 1.04  # Interest rate
    rho: float = 0.9  # AR(1) coefficient for income process
    sigma: float = 0.1  # Std dev of income shocks
    T: int = 100  # Finite horizon for evaluation
    
    def __post_init__(self):
        """Validate parameter ranges."""
        assert 0 <= self.beta < 1, f"beta must be in [0,1), got {self.beta}"
        assert 0 < self.r < 1/self.beta, f"r must be in (0, 1/beta), got {self.r}"
        assert self.sigma > 0, f"sigma must be positive, got {self.sigma}"
        assert abs(self.rho) < 1, f"rho must satisfy |rho| < 1, got {self.rho}"


class ConsumptionSavingModel:
    """
    Consumption-Saving model with borrowing constraint.
    
    Solves:
        max E_0 [sum_t beta^t u(c_t)]
    subject to:
        0 <= c_t <= w_t
        w_{t+1} = r(w_t - c_t) + exp(y_t)
        y_{t+1} = rho*y_t + sigma*epsilon_t, epsilon_t ~ N(0,1)
    """
    
    def __init__(self, params: ConsumptionSavingParams):
        """Initialize the model with given parameters."""
        self.params = params
    
    def utility(self, c: np.ndarray) -> np.ndarray:
        """
        CRRA utility function.
        
        u(c) = (c^(1-gamma) - 1) / (1 - gamma)
        
        Args:
            c: consumption values
            
        Returns:
            utility values
        """
        gamma = self.params.gamma
        if gamma == 1.0:
            return np.log(c)
        else:
            return (c**(1 - gamma) - 1) / (1 - gamma)
    
    def utility_derivative(self, c: np.ndarray) -> np.ndarray:
        """
        Marginal utility (derivative of CRRA).
        
        u'(c) = c^(-gamma)
        
        Args:
            c: consumption values
            
        Returns:
            marginal utility values
        """
        return c**(-self.params.gamma)
    
    def state_transition(
        self,
        w_t: np.ndarray,
        c_t: np.ndarray,
        y_t: np.ndarray
    ) -> np.ndarray:
        """
        State transition for cash-on-hand.
        
        w_{t+1} = r(w_t - c_t) + exp(y_t)
        
        Args:
            w_t: current cash-on-hand
            c_t: current consumption
            y_t: current log-income
            
        Returns:
            w_{t+1} (next period cash-on-hand)
        """
        return self.params.r * (w_t - c_t) + np.exp(y_t)
    
    def income_transition(
        self,
        y_t: np.ndarray,
        eps_t: np.ndarray
    ) -> np.ndarray:
        """
        Income process transition.
        
        y_{t+1} = rho*y_t + sigma*eps_t
        
        Args:
            y_t: current log-income
            eps_t: shocks, N(0,1)
            
        Returns:
            y_{t+1} (next period log-income)
        """
        return self.params.rho * y_t + self.params.sigma * eps_t
    
    def simulate_path(
        self,
        policy_fn,
        y0: np.ndarray,
        w0: np.ndarray,
        T: Optional[int] = None,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a path of the economy.
        
        Args:
            policy_fn: callable(y, w) -> c (consumption policy)
            y0: initial log-income (batch_size,)
            w0: initial cash-on-hand (batch_size,)
            T: time horizon (default: self.params.T)
            rng: random number generator
            
        Returns:
            (y_path, w_path, c_path, u_path) each of shape (T, batch_size)
        """
        if T is None:
            T = self.params.T
        if rng is None:
            rng = np.random.default_rng()
        
        batch_size = y0.shape[0]
        y_path = np.zeros((T, batch_size))
        w_path = np.zeros((T, batch_size))
        c_path = np.zeros((T, batch_size))
        u_path = np.zeros((T, batch_size))
        
        y_t = y0.copy()
        w_t = w0.copy()
        
        for t in range(T):
            # Get consumption from policy
            c_t = policy_fn(y_t, w_t)
            
            # Enforce feasibility
            c_t = np.clip(c_t, 0, w_t)
            
            # Record
            y_path[t] = y_t
            w_path[t] = w_t
            c_path[t] = c_t
            u_path[t] = self.utility(c_t)
            
            # Transition to next period
            eps_next = rng.standard_normal(batch_size)
            w_t = self.state_transition(w_t, c_t, y_t)
            y_t = self.income_transition(y_t, eps_next)
        
        return y_path, w_path, c_path, u_path
    
    def lifetime_reward(self, u_path: np.ndarray) -> np.ndarray:
        """
        Compute lifetime reward (discounted sum of utilities).
        
        LR = sum_t beta^t * u(c_t)
        
        Args:
            u_path: utility path of shape (T, batch_size)
            
        Returns:
            lifetime reward for each batch element (batch_size,)
        """
        T = u_path.shape[0]
        discount_factors = np.array([self.params.beta**t for t in range(T)])
        return np.sum(discount_factors[:, np.newaxis] * u_path, axis=0)
    
    def euler_residual(
        self,
        y_t: np.ndarray,
        w_t: np.ndarray,
        c_t: np.ndarray,
        c_next: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Euler equation residual.
        
        The Euler equation is:
            u'(c_t) = beta * r * E[u'(c_{t+1})]
        
        Unit-free version:
            h = 1 - (beta * r * u'(c_next)) / u'(c_t)
        
        And slackness:
            a = 1 - c_t / w_t
        
        Args:
            y_t: log-income at t
            w_t: cash-on-hand at t
            c_t: consumption at t
            c_next: consumption at t+1
            
        Returns:
            (a, h) unit-free slackness and Euler residual
        """
        u_c_t = self.utility_derivative(c_t)
        u_c_next = self.utility_derivative(c_next)
        
        a = 1.0 - c_t / w_t
        h = 1.0 - (self.params.beta * self.params.r * u_c_next) / u_c_t
        
        return a, h
    
    def fischer_burmeister(self, a: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Fischer-Burmeister function for complementarity conditions.
        
        Psi^FB(a, h) = a + h - sqrt(a^2 + h^2)
        
        Args:
            a: slackness term
            h: Lagrange multiplier term
            
        Returns:
            FB residual
        """
        return a + h - np.sqrt(a**2 + h**2 + 1e-12)  # Add small epsilon for stability
    
    def weighted_fischer_burmeister(
        self,
        a: np.ndarray,
        h: np.ndarray,
        nu: float = 1.0
    ) -> np.ndarray:
        """
        Weighted Fischer-Burmeister function.
        
        Psi^FB(a, nu*h)
        
        Args:
            a: slackness term
            h: Lagrange multiplier term
            nu: weight on h term
            
        Returns:
            weighted FB residual
        """
        return self.fischer_burmeister(a, nu * h)
    
    def create_gauss_hermite_quadrature(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create Gauss-Hermite quadrature nodes and weights.
        
        Args:
            n: number of quadrature nodes
            
        Returns:
            (nodes, weights) for Gauss-Hermite quadrature
        """
        # For Gauss-Hermite: compute roots of Hermite polynomial
        # and corresponding weights
        from numpy.polynomial.hermite import hermgauss
        nodes, weights = hermgauss(n)
        # Transform to standard Normal N(0,1):
        # E[f(eps)] ≈ sum_i w_i f(sqrt(2) * x_i) / sqrt(pi)
        nodes = nodes * np.sqrt(2.0)
        weights = weights / np.sqrt(np.pi)
        return nodes, weights

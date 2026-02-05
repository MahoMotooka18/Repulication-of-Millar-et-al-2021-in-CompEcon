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
    """
    Parameters for the consumption-saving problem (Section 4, Maliar et al. 2021).
    
    Solves: max E_0 [sum_t beta^t u(c_t)]
    subject to: w_{t+1} = r(w_t - c_t) + exp(y_t), 0 <= c_t <= w_t
                y_{t+1} = rho*y_t + sigma*epsilon_t
    
    Attributes:
        gamma: Risk aversion coefficient (CRRA utility parameter)
        beta: Discount factor in [0, 1)
        r: Gross interest rate (return on savings)
        rho: AR(1) coefficient for log-income process
        sigma: Standard deviation of log-income shocks
        T: Time horizon for evaluation
    """
    gamma: float = 2.0
    beta: float = 0.9
    r: float = 1.04
    rho: float = 0.9
    sigma: float = 0.1
    T: int = 100
    
    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        assert 0 <= self.beta < 1, f"beta must be in [0,1), got {self.beta}"
        assert 0 < self.r < 1/self.beta, f"r must be in (0, 1/beta), got {self.r}"
        assert self.sigma > 0, f"sigma must be positive, got {self.sigma}"
        assert abs(self.rho) < 1, f"rho must satisfy |rho| < 1, got {self.rho}"


class ConsumptionSavingModel:
    """
    Consumption-Saving model with borrowing constraint (Section 4 of Maliar et al. 2021).
    
    Implements the agent's optimization problem:
        max E_0 [sum_t beta^t u(c_t)]
    subject to:
        0 <= c_t <= w_t  (borrowing constraint)
        w_{t+1} = r(w_t - c_t) + exp(y_t)  (budget constraint)
        y_{t+1} = rho*y_t + sigma*epsilon_t, epsilon_t ~ N(0,1)  (income process)
    
    This model is used for three training objectives:
    1. Lifetime reward maximization
    2. Euler equation residuals  
    3. Bellman equation residuals
    """
    
    def __init__(self, params: ConsumptionSavingParams) -> None:
        """
        Initialize the consumption-saving model.
        
        Args:
            params: ConsumptionSavingParams instance with model parameters.
        """
        self.params = params
    
    def utility(self, c: np.ndarray) -> np.ndarray:
        """
        Compute CRRA (Constant Relative Risk Aversion) utility.
        
        Utility function: u(c) = (c^(1-gamma) - 1) / (1 - gamma)
        Special case: u(c) = ln(c) when gamma = 1.0
        
        Args:
            c: Consumption array of any shape.
        
        Returns:
            Utility values with the same shape as input.
        """
        gamma = self.params.gamma
        if gamma == 1.0:
            return np.log(c)
        else:
            return (c**(1 - gamma) - 1) / (1 - gamma)
    
    def utility_derivative(self, c: np.ndarray) -> np.ndarray:
        """
        Compute marginal utility (first derivative of CRRA).
        
        Marginal utility: u'(c) = c^(-gamma)
        
        Args:
            c: Consumption array of any shape (must be positive).
        
        Returns:
            Marginal utility values with the same shape as input.
        """
        return c**(-self.params.gamma)
    
    def state_transition(
        self,
        w_t: np.ndarray,
        c_t: np.ndarray,
        y_t: np.ndarray
    ) -> np.ndarray:
        """
        Compute next-period cash-on-hand via budget constraint.
        
        Budget constraint: w_{t+1} = r(w_t - c_t) + exp(y_t)
        
        Args:
            w_t: Current cash-on-hand array.
            c_t: Current consumption array (should satisfy 0 <= c_t <= w_t).
            y_t: Current log-income array.
        
        Returns:
            Next-period cash-on-hand w_{t+1}.
        """
        return self.params.r * (w_t - c_t) + np.exp(y_t)
    
    def income_transition(
        self,
        y_t: np.ndarray,
        eps_t: np.ndarray
    ) -> np.ndarray:
        """
        Compute next-period log-income via AR(1) process.
        
        Income process: y_{t+1} = rho*y_t + sigma*eps_t
        
        Args:
            y_t: Current log-income array.
            eps_t: Standard normal shocks, eps_t ~ N(0, 1).
        
        Returns:
            Next-period log-income y_{t+1}.
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
        Simulate a forward path of the consumption-saving economy.
        
        Given an initial state and a consumption policy, simulate T periods
        forward using the model dynamics and collect consumption and utility.
        
        Args:
            policy_fn: Callable that takes (y_t, w_t) and returns consumption c_t.
            y0: Initial log-income array of shape (batch_size,).
            w0: Initial cash-on-hand array of shape (batch_size,).
            T: Time horizon (default: self.params.T). If None, uses model horizon.
            rng: Random number generator. If None, creates new default_rng().
        
        Returns:
            Tuple of four arrays, each of shape (T, batch_size):
            - y_path: Log-income path
            - w_path: Cash-on-hand path
            - c_path: Consumption path
            - u_path: Utility path
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
            # Evaluate policy function
            c_t = policy_fn(y_t, w_t)
            
            # Enforce borrowing constraint
            c_t = np.clip(c_t, 0, w_t)
            
            # Record state and actions
            y_path[t] = y_t
            w_path[t] = w_t
            c_path[t] = c_t
            u_path[t] = self.utility(c_t)
            
            # Update state for next period
            eps_next = rng.standard_normal(batch_size)
            w_t = self.state_transition(w_t, c_t, y_t)
            y_t = self.income_transition(y_t, eps_next)
        
        return y_path, w_path, c_path, u_path
    
    def lifetime_reward(self, u_path: np.ndarray) -> np.ndarray:
        """
        Compute lifetime reward as discounted sum of utilities.
        
        Lifetime reward: LR = sum_t beta^t * u_t
        
        Args:
            u_path: Utility path of shape (T, batch_size).
        
        Returns:
            Lifetime reward for each sample, shape (batch_size,).
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
        Compute unit-free Euler equation residuals for complementarity.
        
        Euler equation (before unit-free transformation):
            u'(c_t) = beta * r * E[u'(c_{t+1})]
        
        Unit-free slackness and multiplier:
            a = 1 - c_t / w_t  (complementary to Lagrange multiplier)
            h = 1 - (beta*r*u'(c_{t+1})) / u'(c_t)  (Euler residual)
        
        Args:
            y_t: Log-income at current period (unused but included for context).
            w_t: Cash-on-hand at current period.
            c_t: Consumption at current period.
            c_next: Consumption at next period (from policy or expectation).
        
        Returns:
            Tuple (a, h) of unit-free residuals, each of shape matching inputs.
        """
        u_c_t = self.utility_derivative(c_t)
        u_c_next = self.utility_derivative(c_next)
        
        a = 1.0 - c_t / w_t
        h = 1.0 - (self.params.beta * self.params.r * u_c_next) / u_c_t
        
        return a, h
    
    def fischer_burmeister(self, a: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Evaluate Fischer-Burmeister (FB) complementarity function.
        
        FB function: Psi^FB(a, h) = a + h - sqrt(a^2 + h^2)
        
        This function encodes the complementary slackness conditions:
            a >= 0,  h >= 0,  a*h = 0
        More precisely, Psi^FB = 0 iff all three hold.
        
        Args:
            a: Slackness term (non-negative).
            h: Lagrange multiplier term (non-negative).
        
        Returns:
            Fischer-Burmeister residual with same shape as inputs.
        """
        return a + h - np.sqrt(a**2 + h**2 + 1e-12)
    
    def weighted_fischer_burmeister(
        self,
        a: np.ndarray,
        h: np.ndarray,
        nu: float = 1.0
    ) -> np.ndarray:
        """
        Evaluate weighted Fischer-Burmeister function.
        
        Weighted FB: Psi^FB(a, nu*h) = a + nu*h - sqrt(a^2 + (nu*h)^2)
        
        The weight nu controls the relative importance of the multiplier term.
        
        Args:
            a: Slackness term.
            h: Lagrange multiplier term.
            nu: Weight on multiplier term (default 1.0).
        
        Returns:
            Weighted Fischer-Burmeister residual.
        """
        return self.fischer_burmeister(a, nu * h)
    
    def create_gauss_hermite_quadrature(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create Gauss-Hermite quadrature nodes and weights.
        
        Gauss-Hermite quadrature is used for numerical integration of functions
        against the standard normal N(0,1) distribution:
            E[f(X)], X ~ N(0,1) ≈ sum_i w_i * f(x_i)
        
        Args:
            n: Number of quadrature nodes (higher n = higher accuracy).
        
        Returns:
            Tuple (nodes, weights) where:
            - nodes: Quadrature points for standard normal, shape (n,)
            - weights: Integration weights, shape (n,), sum to 1
        """
        from numpy.polynomial.hermite import hermgauss
        nodes, weights = hermgauss(n)
        # Transform from physicist's Hermite to probabilist's (standard normal):
        nodes = nodes * np.sqrt(2.0)
        weights = weights / np.sqrt(np.pi)
        return nodes, weights

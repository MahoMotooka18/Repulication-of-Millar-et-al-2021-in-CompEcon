"""
Neural Network Policy for Krusell-Smith Model

Implements policy networks parameterizing:
1. Consumption share phi(y, w, z; theta)
2. Lagrange multiplier h(y, w, z; theta)
3. Value function V(y, w, z; theta)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict


class KSNeuralNetworkPolicy(nn.Module):
    """
    Neural Network Policy for Krusell-Smith Model (Section 3.2, note.md).
    
    Parameterizes policy functions for heterogeneous agents with distribution features:
    - Consumption share: phi(y_i, w_i, z, D_t; theta) in [0,1]
    - Lagrange multiplier: h(y_i, w_i, z, D_t; theta) >= 0
    - Value function: V(y_i, w_i, z, D_t; theta) unrestricted
    
    Input concatenation:
        [y^i, w^i, z, D_t] where D_t = flattened distribution of agent states
    
    Three output heads with sigmoid, exponential, and linear activations respectively.
    """
    
    def __init__(
        self,
        distribution_features: int,
        hidden_size: int = 64,
        init_intercept_zero: bool = True,
        phi_steady: float | None = None
    ) -> None:
        """
        Initialize the Krusell-Smith policy network.
        
        Args:
            distribution_features: Dimension of distribution features D_t (typically 2*num_agents).
            hidden_size: Number of neurons per hidden layer.
            init_intercept_zero: If True, initialize intercepts to zero.
            phi_steady: Optional steady-state consumption share for initialization.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.distribution_features = distribution_features
        
        # Total input dimension: individual state [y, w, z] + distribution features
        input_dim = 3 + distribution_features
        
        # Activation function (per paper baseline)
        self.activation = torch.sigmoid
        
        # ===== Shared core: 2 hidden layers =====
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # ===== Output heads =====
        # Head 1: Consumption share phi(.) = sigmoid(zeta_0 + eta(...))
        self.phi_intercept = nn.Parameter(torch.zeros(1), requires_grad=False)
        phi_shift = 0.0 if phi_steady is None else self._logit(phi_steady)
        self.register_buffer(
            "phi_logit_shift",
            torch.tensor(phi_shift, dtype=torch.float32)
        )
        self.phi_output = nn.Linear(hidden_size, 1)
        
        # Head 2: Multiplier h(.) = exp(zeta_0 + eta(...))
        self.h_intercept = nn.Parameter(torch.zeros(1))
        self.h_output = nn.Linear(hidden_size, 1)
        
        # Head 3: Value V(.) = zeta_0 + eta(...)
        self.v_intercept = nn.Parameter(torch.zeros(1))
        self.v_output = nn.Linear(hidden_size, 1)
        
        # Initialize network parameters
        self._initialize_weights(init_intercept_zero)
    
    def _initialize_weights(self, init_intercept_zero: bool = True) -> None:
        """
        Initialize network weights and biases.
        
        Per Maliar et al. (2021):
        - Weights: Glorot Uniform (Xavier) initialization
        - Biases: He Uniform initialization
        - Intercepts: zero if init_intercept_zero=True
        
        Args:
            init_intercept_zero: If True, initialize intercepts to zero.
        """
        for layer in [self.fc1, self.fc2, self.phi_output,
                      self.h_output, self.v_output]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                fan_in = layer.weight.shape[1]
                bound = np.sqrt(2.0 / fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)
        
        if init_intercept_zero:
            nn.init.zeros_(self.phi_intercept)
            nn.init.zeros_(self.h_intercept)
            nn.init.zeros_(self.v_intercept)

    @staticmethod
    def _logit(prob: float) -> float:
        """
        Compute stable logit (log-odds) for probability inputs.
        
        logit(p) = log(p/(1-p))
        
        Args:
            prob: Probability value in (0, 1).
        
        Returns:
            Log-odds value.
        """
        p = min(max(prob, 1e-6), 1.0 - 1e-6)
        return float(np.log(p / (1.0 - p)))
    
    def forward_shared(
        self,
        y: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        dist_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through shared neural network core.
        
        Concatenates [y, w, z, D_t] and passes through two hidden layers
        with sigmoid activation to produce eta(y,w,z,D_t;vartheta).
        
        Args:
            y: Idiosyncratic log-productivity, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
            z: Aggregate log-TFP, shape (batch_size,).
            dist_features: Distribution features D_t, shape (batch_size, dist_features).
        
        Returns:
            eta: Shared network output, shape (batch_size, hidden_size).
        """
        x = torch.cat([y.unsqueeze(-1), w.unsqueeze(-1),
                       z.unsqueeze(-1), dist_features], dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x
    
    def forward_phi(
        self,
        y: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        dist_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consumption share phi(y,w,z,D_t;theta) in [0,1].
        
        Parameterization: phi = sigmoid(zeta_0 + eta(y,w,z,D_t;vartheta))
        Applied to determine consumption: c = w * phi, respecting borrowing constraint.
        
        Args:
            y: Idiosyncratic log-productivity, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
            z: Aggregate log-TFP, shape (batch_size,).
            dist_features: Distribution features, shape (batch_size, dist_features).
        
        Returns:
            Consumption share phi, shape (batch_size,), values in [0,1].
        """
        eta = self.forward_shared(y, w, z, dist_features)
        logit = (
            self.phi_intercept
            + self.phi_logit_shift
            + self.phi_output(eta).squeeze(-1)
        )
        return torch.sigmoid(logit)
    
    def forward_h(
        self,
        y: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        dist_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Lagrange multiplier h(y,w,z,D_t;theta) >= 0 for borrowing constraint.
        
        Parameterization: h = exp(zeta_1 + eta(y,w,z,D_t;vartheta))
        Non-negativity enforced by exponential transformation. For interior solutions,
        h=0 (constraint inactive); for boundary solutions, h>0 (constraint binds).
        
        Args:
            y: Idiosyncratic log-productivity, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
            z: Aggregate log-TFP, shape (batch_size,).
            dist_features: Distribution features, shape (batch_size, dist_features).
        
        Returns:
            Lagrange multiplier h >= 0, shape (batch_size,).
        """
        eta = self.forward_shared(y, w, z, dist_features)
        log_h = self.h_intercept + self.h_output(eta).squeeze(-1)
        return torch.exp(log_h)
    
    def forward_v(
        self,
        y: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        dist_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value function approximation V(y,w,z,D_t;theta).
        
        Parameterization: V = zeta_2 + eta(y,w,z,D_t;vartheta)
        Unrestricted output used to approximate the true value function.
        Employed in Bellman objective (Eq. 32) for computing value derivatives
        via finite difference.
        
        Args:
            y: Idiosyncratic log-productivity, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
            z: Aggregate log-TFP, shape (batch_size,).
            dist_features: Distribution features, shape (batch_size, dist_features).
        
        Returns:
            Value function approximation V, shape (batch_size,), unrestricted.
        """
        eta = self.forward_shared(y, w, z, dist_features)
        return self.v_intercept + self.v_output(eta).squeeze(-1)
    
    def forward_policy(
        self,
        y: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        dist_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute optimal consumption c(y,w,z,D_t;theta) satisfying borrowing constraint.
        
        Formula: c = w * phi(y,w,z,D_t;theta) where phi in [0,1]
        This parameterization ensures 0 <= c <= w, respecting the borrowing constraint w >= 0.
        
        Args:
            y: Idiosyncratic log-productivity, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
            z: Aggregate log-TFP, shape (batch_size,).
            dist_features: Distribution features, shape (batch_size, dist_features).
        
        Returns:
            Optimal consumption c, shape (batch_size,).
        """
        phi = self.forward_phi(y, w, z, dist_features)
        return w * phi
    
    def forward_all(
        self,
        y: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        dist_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute all policy outputs in one forward pass.
        
        Returns all four key objects needed by training objectives:
        - Consumption c: respects borrowing constraint via phi in [0,1]
        - Consumption share phi: auxiliary output for multiplicative parameterization
        - Multiplier h: Lagrange multiplier for constraint (h=0 if unconstrained, h>0 if binding)
        - Value V: value function approximation for Bellman equation training
        
        Args:
            y: Idiosyncratic log-productivity, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
            z: Aggregate log-TFP, shape (batch_size,).
            dist_features: Distribution features D_t, shape (batch_size, dist_features).
        
        Returns:
            Tuple of (c, phi, h, V):
            - c: Consumption, shape (batch_size,)
            - phi: Consumption share, shape (batch_size,)
            - h: Lagrange multiplier, shape (batch_size,)
            - V: Value approximation, shape (batch_size,)
        """
        phi = self.forward_phi(y, w, z, dist_features)
        h = self.forward_h(y, w, z, dist_features)
        V = self.forward_v(y, w, z, dist_features)
        c = w * phi
        return c, phi, h, V
    
    def get_distribution_vector(
        self,
        w: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Build the distribution vector D_t = [y_1,...,y_n, w_1,...,w_n].
        
        This concatenates the idiosyncratic state variables for all agents to form
        distribution features passed to the neural network. Used for heterogeneous-agent
        policy computation and AiO operator implementation.
        
        Args:
            w: Cash-on-hand across agents, shape (n_agents,).
            y: Idiosyncratic log-productivity across agents, shape (n_agents,).
        
        Returns:
            Distribution vector D_t, shape (2*n_agents,).
        """
        return np.concatenate([y, w], axis=0)
    
    @property
    def total_params(self) -> int:
        """
        Total number of trainable parameters in the network.
        
        Returns:
            Sum of parameter counts across all neural network layers.
        """
        return sum(p.numel() for p in self.parameters())


class KSPolicyFactory:
    """
    Factory class for creating Krusell-Smith policy networks.
    
    Provides a convenient interface for instantiating policy networks with
    appropriate dimensionality based on model parameters (number of agents).
    """
    
    @staticmethod
    def create_policy(
        hidden_size: int = 64,
        num_agents: int = 1000,
        device: str = 'cpu',
        phi_steady: float | None = None
    ) -> KSNeuralNetworkPolicy:
        """
        Create a Krusell-Smith policy network with specified configuration.
        
        The distribution features dimension is set to 2*num_agents, reflecting
        the concatenation of all agents' productivity and cash-on-hand variables
        to form D_t = [y_1,...,y_n, w_1,...,w_n].
        
        Args:
            hidden_size: Number of neurons in each hidden layer of shared core network.
                Default: 64.
            num_agents: Number of heterogeneous agents in the economy.
                Used to determine distribution_features = 2*num_agents.
                Default: 1000.
            device: Computing device ('cpu' or 'cuda').
                Default: 'cpu'.
            phi_steady: Steady-state consumption share for initialization of phi output.
                If None, initialized from Glorot distribution.
                Default: None.
        
        Returns:
            KSNeuralNetworkPolicy: Policy network configured for heterogeneous-agent
                training with distribution features matching num_agents.
        """
        policy = KSNeuralNetworkPolicy(
            hidden_size=hidden_size,
            distribution_features=2 * num_agents,
            init_intercept_zero=True,
            phi_steady=phi_steady
        )
        return policy.to(device)

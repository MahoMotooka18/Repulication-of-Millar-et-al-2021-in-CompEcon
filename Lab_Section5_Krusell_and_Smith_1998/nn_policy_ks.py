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
    Neural network policy for heterogeneous-agent model.
    
    Inputs: [y_i, w_i, z, D_t]
    where D_t = concatenation of all agents' (y, w) at time t
    
    Outputs:
    - consumption share phi in [0,1]
    - multiplier h >= 0
    - value function V
    """
    
    def __init__(
        self,
        distribution_features: int,
        hidden_size: int = 64,
        init_intercept_zero: bool = True
    ):
    
        """
        Initialize KS policy network.
        
        Args:
            hidden_size: neurons per hidden layer
            distribution_features: length of D_t vector (2 * num_agents)
            init_intercept_zero: initialize intercepts to zero
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.distribution_features = distribution_features
        
        # Input dimension: [y, w, z, dist_features] = 2 + 1 + dist_features
        input_dim = 3 + distribution_features
        
        # Activation (paper baseline: sigmoid)
        self.activation = torch.sigmoid
        
        # Shared core: 2 hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Output heads
        self.phi_intercept = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.phi_output = nn.Linear(hidden_size, 1)
        
        self.h_intercept = nn.Parameter(torch.zeros(1))
        self.h_output = nn.Linear(hidden_size, 1)
        
        self.v_intercept = nn.Parameter(torch.zeros(1))
        self.v_output = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights(init_intercept_zero)
    
    def _initialize_weights(self, init_intercept_zero: bool = True):
        """Initialize using He and Glorot distributions."""
        for layer in [self.fc1, self.fc2, self.phi_output,
                      self.h_output, self.v_output]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                fan_in = layer.weight.shape[1]
                nn.init.uniform_(layer.bias,
                                -np.sqrt(2/fan_in),
                                np.sqrt(2/fan_in))
        
        if init_intercept_zero:
            nn.init.zeros_(self.phi_intercept)
            nn.init.zeros_(self.h_intercept)
            nn.init.zeros_(self.v_intercept)
    
    def forward_shared(
        self,
        y: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        dist_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through shared core.
        
        Args:
            y: idiosyncratic productivity (batch_size,)
            w: cash-on-hand (batch_size,)
            z: aggregate productivity (batch_size,)
            dist_features: distribution features (batch_size, dist_features)
            
        Returns:
            eta: shared network output (batch_size, hidden_size)
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
        Consumption share: phi in [0,1].
        
        phi = sigmoid(zeta_0 + eta(...))
        """
        eta = self.forward_shared(y, w, z, dist_features)
        logit = self.phi_intercept + self.phi_output(eta).squeeze(-1)
        return torch.sigmoid(logit)
    
    def forward_h(
        self,
        y: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        dist_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Multiplier: h >= 0.
        
        h = exp(zeta_0 + eta(...))
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
        Value function: V unrestricted.
        
        V = zeta_0 + eta(...)
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
        """Get consumption from policy: c = w * phi."""
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
        Get all outputs: (c, phi, h, V).
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
        Build the full distribution vector D_t = [y_1..y_n, w_1..w_n].
        """
        return np.concatenate([y, w], axis=0)
    
    @property
    def total_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


class KSPolicyFactory:
    """Factory for creating KS policy networks."""
    
    @staticmethod
    def create_policy(
        hidden_size: int = 64,
        num_agents: int = 1000,
        device: str = 'cpu'
    ) -> KSNeuralNetworkPolicy:
        """
        Create a KS policy network.
        
        Args:
            hidden_size: neurons per hidden layer
            num_agents: number of agents in the economy
            device: 'cpu' or 'cuda'
            
        Returns:
            policy network
        """
        policy = KSNeuralNetworkPolicy(
            hidden_size=hidden_size,
            distribution_features=2 * num_agents,
            init_intercept_zero=True
        )
        return policy.to(device)

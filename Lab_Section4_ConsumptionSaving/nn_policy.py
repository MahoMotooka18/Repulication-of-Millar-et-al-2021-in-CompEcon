"""
Neural Network Policy Representations

Implements three neural network components for solving the consumption-saving problem:
1. Consumption share phi(y,w;theta) bounded in [0,1]
2. Multiplier h(y,w;theta) for Euler/Bellman methods
3. Value function V(y,w;theta) for Bellman method
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict
import numpy as np


class NeuralNetworkPolicy(nn.Module):
    """
    Neural network policy with shared core and three output heads.
    
    Architecture:
    - Shared core: 2 hidden layers with ReLU activation
    - Output 1 (consumption share): sigmoid activation -> phi in [0,1]
    - Output 2 (multiplier): exp activation -> h >= 0
    - Output 3 (value function): linear -> V unrestricted
    """
    
    def __init__(
        self,
        hidden_size: int = 32,
        use_leaky_relu: bool = True,
        leaky_relu_alpha: float = 0.1,
        init_intercept_zero: bool = True
    ):
        """
        Initialize the neural network policy.
        
        Args:
            hidden_size: number of neurons in each hidden layer
            use_leaky_relu: whether to use Leaky ReLU (True) or regular ReLU
            leaky_relu_alpha: alpha parameter for Leaky ReLU
            init_intercept_zero: initialize final intercept to zero
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_leaky_relu = use_leaky_relu
        self.leaky_relu_alpha = leaky_relu_alpha
        
        # Activation function
        if use_leaky_relu:
            self.activation = nn.LeakyReLU(leaky_relu_alpha)
        else:
            self.activation = nn.ReLU()
        
        # Shared core: 2 hidden layers
        # Input: [y, w] (2 features)
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Output heads
        # All have the same structure: core output + intercept
        # For eta(y,w; vartheta), then apply transformations
        
        # Head 1: Consumption share phi(y,w) = sigmoid(zeta_0 + eta(...))
        self.phi_intercept = nn.Parameter(torch.zeros(1))
        self.phi_output = nn.Linear(hidden_size, 1)
        
        # Head 2: Multiplier h(y,w) = exp(zeta_0 + eta(...))
        self.h_intercept = nn.Parameter(torch.zeros(1))
        self.h_output = nn.Linear(hidden_size, 1)
        
        # Head 3: Value V(y,w) = zeta_0 + eta(...)
        self.v_intercept = nn.Parameter(torch.zeros(1))
        self.v_output = nn.Linear(hidden_size, 1)
        
        # Initialize weights and biases
        self._initialize_weights(init_intercept_zero)
    
    def _initialize_weights(self, init_intercept_zero: bool = True):
        """
        Initialize network weights following He/Glorot convention.
        
        Per paper specification:
        - Biases: He Uniform distribution
        - Weights: Glorot Uniform distribution
        """
        for layer in [self.fc1, self.fc2, self.phi_output, self.h_output, self.v_output]:
            # Glorot uniform for weights
            nn.init.xavier_uniform_(layer.weight)
            # He uniform for biases (sqrt(2/fan_in))
            if layer.bias is not None:
                fan_in = layer.weight.shape[1]
                nn.init.uniform_(layer.bias, -np.sqrt(2/fan_in), np.sqrt(2/fan_in))
        
        if init_intercept_zero:
            nn.init.zeros_(self.phi_intercept)
            nn.init.zeros_(self.h_intercept)
            nn.init.zeros_(self.v_intercept)
    
    def forward_shared(self, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared core network.
        
        Args:
            y: log-income (batch_size,)
            w: cash-on-hand (batch_size,)
            
        Returns:
            eta_output: shared network output (batch_size, hidden_size)
        """
        x = torch.stack([y, w], dim=-1)  # (batch_size, 2)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x  # (batch_size, hidden_size)
    
    def forward_phi(self, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Consumption share: phi(y,w) in [0,1].
        
        phi(y,w) = sigmoid(zeta_0 + eta(y,w; vartheta))
        
        Args:
            y: log-income (batch_size,)
            w: cash-on-hand (batch_size,)
            
        Returns:
            phi: consumption share (batch_size,)
        """
        eta = self.forward_shared(y, w)
        logit = self.phi_intercept + self.phi_output(eta).squeeze(-1)
        phi = torch.sigmoid(logit)
        return phi
    
    def forward_h(self, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Unit-free Lagrange multiplier: h(y,w) >= 0.
        
        h(y,w) = exp(zeta_0 + eta(y,w; vartheta))
        
        Args:
            y: log-income (batch_size,)
            w: cash-on-hand (batch_size,)
            
        Returns:
            h: multiplier (batch_size,), always positive
        """
        eta = self.forward_shared(y, w)
        log_h = self.h_intercept + self.h_output(eta).squeeze(-1)
        h = torch.exp(log_h)
        return h
    
    def forward_v(self, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Value function: V(y,w) unrestricted.
        
        V(y,w) = zeta_0 + eta(y,w; vartheta)
        
        Args:
            y: log-income (batch_size,)
            w: cash-on-hand (batch_size,)
            
        Returns:
            V: value (batch_size,)
        """
        eta = self.forward_shared(y, w)
        V = self.v_intercept + self.v_output(eta).squeeze(-1)
        return V
    
    def forward_policy(self, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Get consumption from policy.
        
        c(y,w) = w * phi(y,w)
        
        Args:
            y: log-income (batch_size,)
            w: cash-on-hand (batch_size,)
            
        Returns:
            c: consumption (batch_size,)
        """
        phi = self.forward_phi(y, w)
        c = w * phi
        return c
    
    def forward(
        self,
        y: torch.Tensor,
        w: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass returning requested outputs.
        
        Args:
            y: log-income (batch_size,)
            w: cash-on-hand (batch_size,)
            return_all: if True, return (c, phi, h, V); else just c
            
        Returns:
            If return_all=False: consumption c
            If return_all=True: tuple (c, phi, h, V)
        """
        phi = self.forward_phi(y, w)
        c = w * phi
        
        if return_all:
            h = self.forward_h(y, w)
            V = self.forward_v(y, w)
            return c, phi, h, V
        else:
            return c
    
    def get_all_outputs(
        self,
        y: torch.Tensor,
        w: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get all outputs in a dictionary.
        
        Args:
            y: log-income (batch_size,)
            w: cash-on-hand (batch_size,)
            
        Returns:
            dict with keys: 'c', 'phi', 'h', 'V'
        """
        phi = self.forward_phi(y, w)
        h = self.forward_h(y, w)
        V = self.forward_v(y, w)
        c = w * phi
        
        return {
            'c': c,
            'phi': phi,
            'h': h,
            'V': V
        }
    
    @property
    def total_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


class PolicyFactory:
    """Factory for creating policy networks with different architectures."""
    
    @staticmethod
    def create_policy(
        hidden_size: int = 32,
        use_leaky_relu: bool = True,
        device: str = 'cpu'
    ) -> NeuralNetworkPolicy:
        """
        Create a policy network.
        
        Args:
            hidden_size: number of neurons per hidden layer
            use_leaky_relu: use Leaky ReLU if True, else ReLU
            device: 'cpu' or 'cuda'
            
        Returns:
            policy network on specified device
        """
        policy = NeuralNetworkPolicy(
            hidden_size=hidden_size,
            use_leaky_relu=use_leaky_relu,
            init_intercept_zero=True
        )
        return policy.to(device)
    
    @staticmethod
    def get_network_sizes() -> list:
        """Standard network sizes to evaluate: 8x8, 16x16, 32x32, 64x64."""
        return [8, 16, 32, 64]

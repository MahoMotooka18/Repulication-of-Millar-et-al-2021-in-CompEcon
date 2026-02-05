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
    Neural Network Policy for Consumption-Saving Problem (Section 4, Maliar et al. 2021).
    
    Parameterizes three related objects using a shared neural network core:
    1. Consumption share: phi(y,w;theta) in [0,1], applied as c = w*phi
    2. Lagrange multiplier: h(y,w;theta) >= 0, for Euler/Bellman objectives
    3. Value function: V(y,w;theta), unrestricted, for Bellman objective
    
    Architecture:
    - Shared core: 2 hidden layers with (Leaky)ReLU, input dimension 2 (y, w)
    - Output heads: each applies transformation (sigmoid, exp, or linear)
    - Common intercept zeta_0 followed by network output eta(y,w;vartheta)
    
    Parameterization (per note.md):
    - phi(y,w;theta) = sigmoid(zeta_0 + eta(y,w;vartheta))
    - h(y,w;theta) = exp(zeta_0 + eta(y,w;vartheta))
    - V(y,w;theta) = zeta_0 + eta(y,w;vartheta)
    """
    
    def __init__(
        self,
        hidden_size: int = 32,
        use_leaky_relu: bool = True,
        leaky_relu_alpha: float = 0.1,
        init_intercept_zero: bool = True
    ) -> None:
        """
        Initialize the neural network policy.
        
        Args:
            hidden_size: Number of neurons in each hidden layer (affects capacity).
            use_leaky_relu: If True, use Leaky ReLU; if False, use ReLU.
            leaky_relu_alpha: Slope parameter for Leaky ReLU (ignored if use_leaky_relu=False).
            init_intercept_zero: If True, initialize intercepts (zeta_0) to zero.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_leaky_relu = use_leaky_relu
        self.leaky_relu_alpha = leaky_relu_alpha
        
        # Choose activation function
        if use_leaky_relu:
            self.activation = nn.LeakyReLU(leaky_relu_alpha)
        else:
            self.activation = nn.ReLU()
        
        # ===== Shared Core Network =====
        # Input: [y, w] (2 features for log-income and cash-on-hand)
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # ===== Output Heads (three independent functions) =====
        # Each head: intercept (zeta_0) + output layer (eta)
        
        # Head 1: Consumption share phi(y,w) = sigmoid(zeta_0 + eta(...))
        self.phi_intercept = nn.Parameter(torch.zeros(1))
        self.phi_output = nn.Linear(hidden_size, 1)
        
        # Head 2: Multiplier h(y,w) = exp(zeta_0 + eta(...))
        self.h_intercept = nn.Parameter(torch.zeros(1))
        self.h_output = nn.Linear(hidden_size, 1)
        
        # Head 3: Value function V(y,w) = zeta_0 + eta(...)
        self.v_intercept = nn.Parameter(torch.zeros(1))
        self.v_output = nn.Linear(hidden_size, 1)
        
        # Initialize weights according to paper specifications
        self._initialize_weights(init_intercept_zero)
    
    def _initialize_weights(self, init_intercept_zero: bool = True) -> None:
        """
        Initialize network weights and biases.
        
        Per Maliar et al. (2021):
        - Weights: Glorot Uniform (Xavier) initialization
        - Biases: He Uniform initialization
        - Intercepts (zeta_0): zero if init_intercept_zero=True
        
        Args:
            init_intercept_zero: If True, set intercepts to zero.
        """
        for layer in [self.fc1, self.fc2, self.phi_output, self.h_output, self.v_output]:
            # Glorot uniform for weights
            nn.init.xavier_uniform_(layer.weight)
            # He uniform for biases: sample from [-sqrt(2/fan_in), +sqrt(2/fan_in)]
            if layer.bias is not None:
                fan_in = layer.weight.shape[1]
                bound = np.sqrt(2.0 / fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)
        
        if init_intercept_zero:
            nn.init.zeros_(self.phi_intercept)
            nn.init.zeros_(self.h_intercept)
            nn.init.zeros_(self.v_intercept)
    
    def forward_shared(self, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the shared neural network core.
        
        Computes eta(y,w;vartheta) where vartheta are the core network parameters.
        Input: [y, w] -> fc1 -> activation -> fc2 -> activation -> output
        
        Args:
            y: Log-income, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
        
        Returns:
            eta output from shared core, shape (batch_size, hidden_size).
        """
        x = torch.stack([y, w], dim=-1)  # Stack to (batch_size, 2)
        x = self.activation(self.fc1(x))  # First hidden layer
        x = self.activation(self.fc2(x))  # Second hidden layer
        return x
    
    def forward_phi(self, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Compute consumption share phi(y,w;theta) in [0,1].
        
        Consumption share parameterization:
            phi(y,w;theta) = sigmoid(zeta_0 + eta(y,w;vartheta))
        
        This is then multiplied by wealth to get consumption: c = w*phi.
        The sigmoid ensures phi ∈ [0,1], respecting the borrowing constraint c ≤ w.
        
        Args:
            y: Log-income, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
        
        Returns:
            Consumption share phi, shape (batch_size,), values in [0,1].
        """
        eta = self.forward_shared(y, w)
        logit = self.phi_intercept + self.phi_output(eta).squeeze(-1)
        phi = torch.sigmoid(logit)
        return phi
    
    def forward_h(self, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Compute unit-free Lagrange multiplier h(y,w;theta) >= 0.
        
        Multiplier parameterization:
            h(y,w;theta) = exp(zeta_0 + eta(y,w;vartheta))
        
        Used in Euler and Bellman objectives to encode the Kuhn-Tucker conditions.
        The exponential ensures h >= 0, which is required by the complementarity conditions.
        
        Args:
            y: Log-income, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
        
        Returns:
            Lagrange multiplier h, shape (batch_size,), always positive.
        """
        eta = self.forward_shared(y, w)
        log_h = self.h_intercept + self.h_output(eta).squeeze(-1)
        h = torch.exp(log_h)
        return h
    
    def forward_v(self, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Compute value function V(y,w;theta) (unrestricted).
        
        Value function parameterization:
            V(y,w;theta) = zeta_0 + eta(y,w;vartheta)
        
        Used in the Bellman objective to minimize the Bellman residual:
            R(y,w) = V(y,w) - u(c) - beta*V(y',w')
        
        The linear activation preserves the sign of V, allowing it to match
        both positive and negative utility values.
        
        Args:
            y: Log-income, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
        
        Returns:
            Value function V, shape (batch_size,), unrestricted range.
        """
        eta = self.forward_shared(y, w)
        V = self.v_intercept + self.v_output(eta).squeeze(-1)
        return V
    
    def forward_policy(self, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Compute consumption policy from wealth and consumption share.
        
        Consumption policy:
            c(y,w;theta) = w * phi(y,w;theta)
        
        This combines the consumption share (bounded in [0,1]) with current
        wealth to produce a feasible consumption level satisfying 0 <= c <= w.
        
        Args:
            y: Log-income, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
        
        Returns:
            Consumption c, shape (batch_size,).
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
        Forward pass returning consumption and optionally other outputs.
        
        Args:
            y: Log-income, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
            return_all: If False, return only consumption c.
                       If True, return tuple (c, phi, h, V).
        
        Returns:
            If return_all=False: consumption c, shape (batch_size,)
            If return_all=True: tuple (c, phi, h, V), each shape (batch_size,)
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
        Get all outputs in a dictionary for easy access.
        
        Args:
            y: Log-income, shape (batch_size,).
            w: Cash-on-hand, shape (batch_size,).
        
        Returns:
            Dictionary with keys:
            - 'c': Consumption
            - 'phi': Consumption share
            - 'h': Lagrange multiplier
            - 'V': Value function
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
        """
        Count the total number of trainable parameters in the network.
        
        Returns:
            Total number of elements across all parameters.
        """
        return sum(p.numel() for p in self.parameters())


class PolicyFactory:
    """Factory class for creating and managing policy networks."""
    
    @staticmethod
    def create_policy(
        hidden_size: int = 32,
        use_leaky_relu: bool = True,
        device: str = 'cpu'
    ) -> NeuralNetworkPolicy:
        """
        Create a neural network policy and move to specified device.
        
        Args:
            hidden_size: Number of neurons per hidden layer.
            use_leaky_relu: If True, use Leaky ReLU; otherwise regular ReLU.
            device: Device to place the network on ('cpu' or 'cuda').
        
        Returns:
            NeuralNetworkPolicy instance on the specified device.
        """
        policy = NeuralNetworkPolicy(
            hidden_size=hidden_size,
            use_leaky_relu=use_leaky_relu,
            init_intercept_zero=True
        )
        return policy.to(device)
    
    @staticmethod
    def get_network_sizes() -> list:
        """
        Get standard network sizes for grid search in experiments.
        
        Per note.md: Compare network architectures with widths 8x8, 16x16, 32x32, 64x64.
        
        Returns:
            List of hidden layer sizes to evaluate: [8, 16, 32, 64]
        """
        return [8, 16, 32, 64]

"""
Policy utilities for Section 5 (Krusell-Smith).

Centralizes policy output definitions and normalization handling.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import numpy as np
import torch


class PolicyOutputType:
    """
    Enumeration of supported policy output parameterizations.
    
    Defines how the neural network outputs are mapped to economic decisions:
    - C_LEVEL: Raw consumption level from network
    - C_SHARE: Consumption as fraction of wealth (implemented via sigmoid phi),
              ensuring 0 ≤ c ≤ w
    """

    C_LEVEL = "c_level"
    C_SHARE = "c_share"


@dataclass
class NormalizationSpec:
    """
    Specifies how wealth and consumption are normalized for the policy network.
    
    Allows flexible scaling and shifting of inputs/outputs to improve numerical
    stability and accelerate training. Different normalizations can be applied to
    inputs (y, w, z) and outputs (c, lambda).
    
    Attributes:
        w_scale: Scaling factor for wealth input normalization.
        w_shift: Shift (offset) applied after wealth scaling.
        w_normalized: Whether to apply wealth normalization.
        c_scale: Scaling factor for consumption output de-normalization.
        c_shift: Shift applied to consumption after scaling.
        c_normalized: Whether output consumption is normalized.
    """

    w_scale: float = 1.0
    w_shift: float = 0.0
    w_normalized: bool = False
    c_scale: float = 1.0
    c_shift: float = 0.0
    c_normalized: bool = False

    def to_dict(self) -> Dict:
        """
        Convert to JSON-serializable dictionary.
        
        Returns:
            Dict: Dataclass fields as dictionary for config serialization.
        """
        return asdict(self)


@dataclass
class InputScaleSpec:
    """
    Input scaling specification for heterogeneous-agent policy networks.
    
    Applies optional normalization to productivity (y), aggregate TFP (z),
    and wealth (w) inputs before passing to neural network. Wealth scaling
    uses min-max normalization to map to [-1, 1] range for numerical stability.
    
    This specification is critical for:
    - Normalizing inputs to similar scales for neural network training
    - Handling non-stationary distributions in heterogeneous-agent models
    - Log-scaling wealth in calibrations where w ranges over many orders of magnitude
    
    Attributes:
        y_scale: Normalization scale for idiosyncratic productivity input.
        z_scale: Normalization scale for aggregate log-TFP input.
        w_min: Minimum wealth for scaling bounds (clamps negative wealth).
        w_max: Maximum wealth for scaling bounds (maps to +1 after normalization).
        w_steady: Steady-state wealth reference level.
        enabled: Whether scaling is active (if False, inputs pass through unchanged).
    """

    y_scale: float = 1.0
    z_scale: float = 1.0
    w_min: float = 0.0
    w_max: float = 1.0
    w_steady: float = 1.0
    enabled: bool = True

    def to_dict(self) -> Dict:
        """
        Convert to JSON-serializable dictionary.
        
        Returns:
            Dict: Dataclass fields as dictionary for config persistence.
        """
        return asdict(self)


def _safe_scale(value: float) -> float:
    """
    Safely divide by a value, returning 1.0 if value is too close to zero.
    
    Prevents numerical instability from division by very small denominators
    in scaling operations. Used to ensure denominators in min-max scaling
    and other normalization operations don't cause inf/nan values.
    
    Args:
        value: Potential denominator value.
    
    Returns:
        float: value if |value| > 1e-12, else 1.0 (neutral scaling factor).
    """
    return value if abs(value) > 1e-12 else 1.0


def scale_inputs_numpy(
    y: np.ndarray,
    w: np.ndarray,
    z: float,
    spec: InputScaleSpec
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Scale heterogeneous-agent state variables for neural network input.
    
    Applies normalization to productivity, wealth, and aggregate TFP according
    to InputScaleSpec. Wealth undergoes min-max normalization to [-1, 1] range.
    Other inputs are simply divided by their scale factors.
    
    Normalization formula for wealth:
        w_norm = (w - w_min) / (w_max - w_min) * 2 - 1
    Maps [w_min, w_max] → [-1, 1]
    
    Args:
        y: Idiosyncratic log-productivity array, shape (num_agents,).
        w: Wealth array, shape (num_agents,).
        z: Aggregate log-TFP scalar.
        spec: InputScaleSpec with normalization parameters.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - y_scaled: Scaled productivity (divide by y_scale)
            - w_scaled: Min-max normalized wealth to [-1, 1] range
            - z_scaled: Scaled TFP (divide by z_scale)
    """
    if not spec.enabled:
        return y, w, z
    y_scale = _safe_scale(spec.y_scale)
    z_scale = _safe_scale(spec.z_scale)
    w_denom = _safe_scale(spec.w_max - spec.w_min)
    w_norm = (w - spec.w_min) / w_denom * 2.0 - 1.0
    return y / y_scale, w_norm, z / z_scale


def scale_inputs_torch(
    y: torch.Tensor,
    w: torch.Tensor,
    z: torch.Tensor,
    spec: InputScaleSpec
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Scale PyTorch state tensors for policy network input (GPU-compatible).
    
    Same normalization as scale_inputs_numpy but operating on torch.Tensor
    objects for GPU acceleration. Enables fast scaling of large batches
    during training and evaluation.
    
    Args:
        y: Productivity tensor, shape (batch_size, num_agents).
        w: Wealth tensor, shape (batch_size, num_agents).
        z: Aggregate log-TFP tensor, shape (batch_size,).
        spec: InputScaleSpec with normalization parameters.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - y_scaled: Normalized productivity
            - w_scaled: Min-max normalized wealth to [-1, 1]
            - z_scaled: Normalized TFP
    """
    if not spec.enabled:
        return y, w, z
    y_scale = float(_safe_scale(spec.y_scale))
    z_scale = float(_safe_scale(spec.z_scale))
    w_denom = float(_safe_scale(spec.w_max - spec.w_min))
    w_norm = (w - spec.w_min) / w_denom * 2.0 - 1.0
    return y / y_scale, w_norm, z / z_scale


def build_dist_features_numpy(y_scaled: np.ndarray, w_scaled: np.ndarray) -> np.ndarray:
    """
    Build normalized distribution features D_t = [y_1,...,y_n, w_1,...,w_n].
    
    Concatenates agent-level state variables into a distribution vector that
    captures the cross-sectional distribution of productivity and wealth.
    Passed to neural network to allow policy to respond to aggregate inequality
    and composition of the wealth distribution (critical for heterogeneous-agent
    modeling and AiO operator).
    
    Args:
        y_scaled: Scaled idiosyncratic productivity array, shape (num_agents,).
        w_scaled: Scaled wealth array (from min-max normalization), shape (num_agents,).
    
    Returns:
        np.ndarray: Distribution vector D_t, shape (2*num_agents,).
    """
    return np.concatenate([y_scaled, w_scaled], axis=0)


def build_dist_features_torch(y_scaled: torch.Tensor, w_scaled: torch.Tensor) -> torch.Tensor:
    """
    Build normalized distribution features D_t (GPU-compatible PyTorch version).
    
    Same functionality as build_dist_features_numpy but operates on torch tensors
    for efficient GPU batch processing during training and evaluation.
    
    Args:
        y_scaled: Scaled productivity tensor, shape (batch_size, num_agents).
        w_scaled: Scaled wealth tensor, shape (batch_size, num_agents).
    
    Returns:
        torch.Tensor: Distribution tensor, shape (batch_size, 2*num_agents).
    """
    return torch.cat([y_scaled, w_scaled], dim=0)


def consumption_from_share_torch(
    policy,
    y_scaled: torch.Tensor,
    w_scaled: torch.Tensor,
    z_scaled: torch.Tensor,
    dist_scaled: torch.Tensor,
    w_raw: torch.Tensor,
    w_cap: float | None = None
) -> torch.Tensor:
    """
    Compute actual consumption from policy consumption share and wealth.
    
    Extracts consumption share φ ∈ [0,1] from policy network and applies:
        c = w_raw × φ
    
    Optional wealth cap (w_cap) handles calibrations with log-scaled bounds,
    enforcing maximum saving constraint on capital accumulation.
    
    Args:
        policy: KSNeuralNetworkPolicy with forward_phi method.
        y_scaled, w_scaled, z_scaled: Scaled state variables for policy input.
        dist_scaled: Distribution features D_t for policy input.
        w_raw: Raw (unscaled) wealth for consumption calculation, shape (batch_size, num_agents).
        w_cap: Optional maximum wealth cap for constraint enforcement.
               If None, no cap applied (unbounded saving).
    
    Returns:
        torch.Tensor: Consumption c_t, shape matching w_raw.
    """
    share = policy.forward_phi(y_scaled, w_scaled, z_scaled, dist_scaled)
    if w_cap is None:
        return share * w_raw
    k_next = torch.minimum(w_raw * (1.0 - share), w_raw.new_full((), float(w_cap)))
    return w_raw - k_next


def normalize_w(w_raw: np.ndarray, spec: NormalizationSpec) -> np.ndarray:
    """
    Normalize raw wealth using multiplicative/additive specification.
    
    Applies affine transformation: w_norm = (w_raw - shift) / scale
    Only applied if spec.w_normalized is True.
    
    Args:
        w_raw: Raw wealth values.
        spec: NormalizationSpec with w_scale and w_shift parameters.
    
    Returns:
        np.ndarray: Normalized wealth (or raw if w_normalized=False).
    """
    if not spec.w_normalized:
        return w_raw
    if spec.w_scale == 0:
        return w_raw
    return (w_raw - spec.w_shift) / spec.w_scale


def unnormalize_w(w_norm: np.ndarray, spec: NormalizationSpec) -> np.ndarray:
    """
    Recover raw wealth from normalized representation.
    
    Reverses affine transformation: w_raw = w_norm × scale + shift
    Inverse of normalize_w. Only applied if spec.w_normalized is True.
    
    Args:
        w_norm: Normalized wealth values.
        spec: NormalizationSpec with w_scale and w_shift parameters.
    
    Returns:
        np.ndarray: Raw wealth (or input if w_normalized=False).
    """
    if not spec.w_normalized:
        return w_norm
    return w_norm * spec.w_scale + spec.w_shift


def unnormalize_c(c_norm: np.ndarray, spec: NormalizationSpec) -> np.ndarray:
    """
    Recover raw consumption from normalized policy output.
    
    Reverses affine transformation applied during policy training:
    c_raw = c_norm × scale + shift
    
    Args:
        c_norm: Normalized (policy) consumption values.
        spec: NormalizationSpec with c_scale and c_shift parameters.
    
    Returns:
        np.ndarray: Raw consumption (or input if c_normalized=False).
    """
    if not spec.c_normalized:
        return c_norm
    return c_norm * spec.c_scale + spec.c_shift


def reconstruct_consumption_level(
    policy_output: np.ndarray,
    w_raw: np.ndarray,
    output_type: str,
    normalization_spec: NormalizationSpec
) -> np.ndarray:
    """
    Reconstruct consumption in raw economic units from policy network output.
    
    Handles conversion based on how policy parameterizes consumption:
    - C_SHARE: c = φ × w where φ ∈ [0,1] is output by network
    - C_LEVEL: c is direct output, may need unnormalization
    
    Args:
        policy_output: Raw network output for consumption (φ or c_norm).
        w_raw: Raw wealth for share-based reconstruction.
        output_type: Either PolicyOutputType.C_SHARE or C_LEVEL.
        normalization_spec: Unnormalization parameters if output_type=C_LEVEL.
    
    Returns:
        np.ndarray: Consumption c in raw scale matching w_raw.
    
    Raises:
        ValueError: If output_type is not recognized.
    """
    if output_type == PolicyOutputType.C_SHARE:
        return policy_output * w_raw
    if output_type == PolicyOutputType.C_LEVEL:
        return unnormalize_c(policy_output, normalization_spec)
    raise ValueError(f"Unknown output_type: {output_type}")


def resolve_policy_output_type(
    objective_name: str,
    config_mapping: Dict[str, str],
    default_type: str
) -> str:
    """
    Resolve the appropriate policy output type for a given training objective.
    
    Maps objective names to their corresponding policy parameterizations,
    enabling flexible switching between consumption share and consumption level
    representations based on training approach.
    
    Args:
        objective_name: Name of training objective (e.g., 'lifetime_reward', 'euler').
        config_mapping: Dict mapping objective names to PolicyOutputType values.
        default_type: Default type if objective_name not in config_mapping.
    
    Returns:
        str: Validated policy output type (C_SHARE or C_LEVEL).
    
    Raises:
        ValueError: If resolved output type is invalid/unrecognized.
    """
    output_type = config_mapping.get(objective_name, default_type)
    if output_type not in (PolicyOutputType.C_LEVEL, PolicyOutputType.C_SHARE):
        raise ValueError(f"Invalid output type: {output_type}")
    return output_type

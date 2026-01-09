"""
Policy utilities for Section 5 (Krusell-Smith).

Centralizes policy output definitions and normalization handling.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import numpy as np
import torch


class PolicyOutputType:
    """Enum-like container for policy output types."""

    C_LEVEL = "c_level"
    C_SHARE = "c_share"


@dataclass
class NormalizationSpec:
    """Normalization specification for wealth and consumption."""

    w_scale: float = 1.0
    w_shift: float = 0.0
    w_normalized: bool = False
    c_scale: float = 1.0
    c_shift: float = 0.0
    c_normalized: bool = False

    def to_dict(self) -> Dict:
        """Return a JSON-serializable dict."""
        return asdict(self)


@dataclass
class InputScaleSpec:
    """Scaling specification for policy inputs."""

    y_scale: float = 1.0
    z_scale: float = 1.0
    w_min: float = 0.0
    w_max: float = 1.0
    w_steady: float = 1.0
    enabled: bool = True

    def to_dict(self) -> Dict:
        """Return a JSON-serializable dict."""
        return asdict(self)


def _safe_scale(value: float) -> float:
    """Avoid division by zero for scaling."""
    return value if abs(value) > 1e-12 else 1.0


def scale_inputs_numpy(
    y: np.ndarray,
    w: np.ndarray,
    z: float,
    spec: InputScaleSpec
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scale numpy inputs for policy evaluation."""
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
    """Scale torch inputs for policy evaluation."""
    if not spec.enabled:
        return y, w, z
    y_scale = float(_safe_scale(spec.y_scale))
    z_scale = float(_safe_scale(spec.z_scale))
    w_denom = float(_safe_scale(spec.w_max - spec.w_min))
    w_norm = (w - spec.w_min) / w_denom * 2.0 - 1.0
    return y / y_scale, w_norm, z / z_scale


def build_dist_features_numpy(y_scaled: np.ndarray, w_scaled: np.ndarray) -> np.ndarray:
    """Build scaled distribution vector D_t."""
    return np.concatenate([y_scaled, w_scaled], axis=0)


def build_dist_features_torch(y_scaled: torch.Tensor, w_scaled: torch.Tensor) -> torch.Tensor:
    """Build scaled distribution vector D_t (torch)."""
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
    """Compute raw consumption from share using scaled inputs."""
    share = policy.forward_phi(y_scaled, w_scaled, z_scaled, dist_scaled)
    if w_cap is None:
        return share * w_raw
    k_next = torch.minimum(w_raw * (1.0 - share), w_raw.new_full((), float(w_cap)))
    return w_raw - k_next


def normalize_w(w_raw: np.ndarray, spec: NormalizationSpec) -> np.ndarray:
    """Normalize raw wealth if configured."""
    if not spec.w_normalized:
        return w_raw
    if spec.w_scale == 0:
        return w_raw
    return (w_raw - spec.w_shift) / spec.w_scale


def unnormalize_w(w_norm: np.ndarray, spec: NormalizationSpec) -> np.ndarray:
    """Unnormalize wealth if configured."""
    if not spec.w_normalized:
        return w_norm
    return w_norm * spec.w_scale + spec.w_shift


def unnormalize_c(c_norm: np.ndarray, spec: NormalizationSpec) -> np.ndarray:
    """Unnormalize consumption if configured."""
    if not spec.c_normalized:
        return c_norm
    return c_norm * spec.c_scale + spec.c_shift


def reconstruct_consumption_level(
    policy_output: np.ndarray,
    w_raw: np.ndarray,
    output_type: str,
    normalization_spec: NormalizationSpec
) -> np.ndarray:
    """Reconstruct consumption level c in raw scale."""
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
    """Resolve the policy output type for a given objective."""
    output_type = config_mapping.get(objective_name, default_type)
    if output_type not in (PolicyOutputType.C_LEVEL, PolicyOutputType.C_SHARE):
        raise ValueError(f"Invalid output type: {output_type}")
    return output_type

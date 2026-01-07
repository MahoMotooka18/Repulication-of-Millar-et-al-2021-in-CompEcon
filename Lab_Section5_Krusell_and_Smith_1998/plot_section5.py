"""
Plot Generation for Section 5 (Krusell-Smith)

Generates publication-quality plots for training and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from policy_utils_ks import (
    PolicyOutputType,
    NormalizationSpec,
    InputScaleSpec,
    normalize_w,
    reconstruct_consumption_level,
    scale_inputs_numpy,
    build_dist_features_numpy,
)


class KSPlotter:
    """Generates plots for KS model results."""

    def __init__(self, figsize: tuple = (15, 5)):
        """
        Initialize plotter.
        
        Args:
            figsize: figure size (width, height)
        """
        self.figsize = figsize

    def _moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        """Compute a simple moving average with full-length output."""
        if window <= 1:
            return values
        series = pd.Series(values)
        return series.rolling(window, min_periods=1).mean().to_numpy()

    def plot_objective_results(
        self,
        metrics_df: pd.DataFrame,
        simulation_dict: Dict,
        objective_name: str,
        output_dir: Path,
        policy=None,
        model=None,
        num_agents: int = 1000,
        use_log_x: bool = True,
        smoothing_window: int = 1,
        show_raw: bool = True,
        policy_output_type: str = PolicyOutputType.C_SHARE,
        normalization_spec: Optional[NormalizationSpec] = None,
        input_scale_spec: Optional[InputScaleSpec] = None,
        mismatch_config: Optional[Dict] = None,
        w_plot_max: Optional[float] = None,
        loss_scale_ref_points: Optional[float] = 10.0
    ):
        """
        Create 3-panel plot for objective function.
        
        Panel 1: Training loss
        Panel 2: Consumption policy (c vs w at different productivity levels)
        Panel 3: Wealth simulation
        
        Args:
            metrics_df: metrics dataframe
            simulation_dict: simulation results
            objective_name: name of objective
            output_dir: output directory
            policy: neural network policy (optional)
            model: KS model (optional)
            num_agents: number of agents
            use_log_x: use log scale on x-axis
        """
        if objective_name == 'bellman':
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        else:
            fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        debug_payload = {}
        mismatch_checks: List[Dict] = []
        
        # Panel 1: Training loss
        ax = axes[0]
        if 'objective_train' in metrics_df.columns:
            losses = metrics_df['objective_train'].values
            if 'update_step' in metrics_df.columns:
                epochs = metrics_df['update_step'].values
                x_label = 'Update'
            else:
                epochs = np.arange(1, len(losses) + 1)
                x_label = 'Update'
            if objective_name == 'lifetime_reward':
                losses = np.abs(losses)
                if ('train_points' in metrics_df.columns
                        and loss_scale_ref_points):
                    losses = (
                        losses *
                        metrics_df['train_points'].values /
                        float(loss_scale_ref_points)
                    )
            if objective_name in ('euler', 'bellman'):
                losses = np.clip(losses, 1e-12, None)
            if show_raw:
                ax.plot(epochs, losses, 'b-', linewidth=1, alpha=0.3)
            smoothed = self._moving_average(losses, smoothing_window)
            ax.plot(epochs, smoothed, 'b-', linewidth=2)
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel('Training Loss', fontsize=12)
            if objective_name in ('euler', 'bellman'):
                ax.set_title('Panel 1: Log losses', fontsize=12)
            else:
                ax.set_title('Panel 1: Training Loss', fontsize=12)
            if use_log_x:
                ax.set_xscale('log')
            if len(epochs) > 0:
                ax.set_xlim(left=max(1.0, float(np.min(epochs))),
                            right=float(np.max(epochs)))
            if objective_name in ('euler', 'bellman'):
                ax.set_yscale('log')
            max_epoch = float(np.max(epochs)) if len(epochs) else 1.0
            xticks = [1.0, 1e1, 1e2, 1e3, 1e4]
            ax.set_xticks([tick for tick in xticks if tick <= max_epoch])
        ax.grid(True, alpha=0.3)

        # Panel 2 (bellman): Value function
        if objective_name == 'bellman':
            ax = axes[1]
            if policy is not None and model is not None:
                normalization_spec = normalization_spec or NormalizationSpec()
                value_plot = self._compute_value_function(
                    policy,
                    model,
                    simulation_dict,
                    normalization_spec,
                    input_scale_spec or InputScaleSpec(),
                    w_plot_max=w_plot_max
                )
                w_grid_raw = value_plot['w_grid_raw']
                y_levels = value_plot['y_grid_raw']
                y_labels = value_plot['y_labels']
                v_grid_raw = value_plot['v_grid_raw']
                colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(y_levels)))
                for idx, y_level in enumerate(y_levels):
                    ax.plot(
                        w_grid_raw,
                        v_grid_raw[idx],
                        color=colors[idx],
                        label=y_labels[idx]
                    )
            ax.set_xlabel('Wealth w', fontsize=12)
            ax.set_ylabel('Value V', fontsize=12)
            ax.set_title('Panel 2: Value Function', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            consumption_ax = axes[2]
        else:
            consumption_ax = axes[1]

        # Consumption policy (c vs w for different y levels)
        ax = consumption_ax
        w_path = simulation_dict['w_path']
        c_path = simulation_dict['c_path']
        y_path = simulation_dict['y_path']

        if policy is not None and model is not None:
            normalization_spec = normalization_spec or NormalizationSpec()
            policy_plot = self._compute_consumption_rule(
                policy,
                model,
                simulation_dict,
                objective_name,
                policy_output_type,
                normalization_spec,
                input_scale_spec or InputScaleSpec(),
                w_plot_max=w_plot_max
            )
            w_grid_raw = policy_plot['w_grid_raw']
            y_levels = policy_plot['y_grid_raw']
            y_labels = policy_plot['y_labels']
            c_grid_raw = policy_plot['c_grid_raw']
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(y_levels)))

            for idx, y_level in enumerate(y_levels):
                ax.plot(
                    w_grid_raw,
                    c_grid_raw[idx],
                    color=colors[idx],
                    label=y_labels[idx]
                )

            debug_payload.update(policy_plot)
            mismatch_checks = self._compute_mismatch_checks(
                objective_name,
                policy_plot,
                simulation_dict,
                mismatch_config or {}
            )
        else:
            y_levels = np.percentile(y_path, [10, 30, 50, 70, 90])
            colors = plt.cm.tab10(np.linspace(0, 1, len(y_levels)))

            for idx, y_level in enumerate(y_levels):
                mask = np.abs(y_path[:, 0] - y_level) < 0.5
                if np.any(mask):
                    w_slice = w_path[mask, 0]
                    c_slice = c_path[mask, 0]
                    ax.scatter(w_slice, c_slice, alpha=0.3, s=5,
                              color=colors[idx],
                              label=f'y={y_level:.1f}')

        ax.set_xlabel('Wealth w', fontsize=12)
        ax.set_ylabel('Consumption c', fontsize=12)
        if objective_name == 'bellman':
            ax.set_title('Panel 3: Consumption Policy', fontsize=12)
        else:
            ax.set_title('Panel 2: Consumption Policy', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Wealth simulation (random agents)
        ax = axes[-1]
        num_agents_to_plot = min(5, w_path.shape[1])
        rng = np.random.default_rng(0)
        agent_idx = rng.choice(w_path.shape[1], size=num_agents_to_plot, replace=False)
        colors_agents = plt.cm.tab10(np.linspace(0, 1, num_agents_to_plot))

        for i, idx in enumerate(agent_idx):
            w_agent = w_path[:min(200, len(w_path)), idx]
            ax.plot(w_agent, label=f'Agent {idx+1}',
                   linewidth=1.5, color=colors_agents[i])
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Wealth', fontsize=12)
        if objective_name == 'bellman':
            ax.set_title('Panel 4: Individual Wealth Simulation', fontsize=12)
        else:
            ax.set_title('Panel 3: Individual Wealth Simulation', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / f"{objective_name}_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {output_path}")
        return {
            'policy_plot': debug_payload,
            'mismatch_checks': mismatch_checks
        }

    def plot_comparison_across_objectives(
        self,
        simulations: Dict[str, Dict],
        output_dir: Path,
        policies: Dict[str, object] = None,
        model=None,
        base_state: Dict = None,
        policy_output_types: Optional[Dict[str, str]] = None,
        normalization_spec: Optional[NormalizationSpec] = None,
        input_scale_spec: Optional[InputScaleSpec] = None,
        w_plot_max: Optional[float] = None
    ):
        """
        Create comparison plot across objectives (Fig. 13 style).
        
        Panel 1: Consumption rule comparison (representative agent)
        Panel 2: Individual wealth simulation (same agent, common shocks)
        Panel 3: Aggregate capital simulation
        
        Args:
            simulations: dict mapping objective -> simulation results
            output_dir: output directory
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        colors = {'lifetime_reward': 'b', 'euler': 'r', 'bellman': 'g'}
        styles = {'lifetime_reward': '-', 'euler': '--', 'bellman': ':'}
        
        # Panel 1: Consumption rule for representative agent
        ax = axes[0]
        
        if policies is not None and model is not None and base_state is not None:
            import torch

            normalization_spec = normalization_spec or NormalizationSpec()
            input_scale_spec = input_scale_spec or InputScaleSpec()
            policy_output_types = policy_output_types or {}

            w_init = base_state['w_init']
            z_init = base_state['z_init']
            num_agents = len(w_init)
            steady_y = 0.0
            if input_scale_spec is not None and input_scale_spec.enabled:
                steady_w = float(input_scale_spec.w_steady)
                w_min = float(input_scale_spec.w_min)
                w_max = float(input_scale_spec.w_max)
            else:
                steady_w = float(np.mean(w_init))
                w_min = 0.0
                w_max = float(np.max(w_init))
            if w_plot_max is not None:
                w_max = min(w_max, float(w_plot_max))
            w_grid = np.linspace(w_min, w_max, 200)

            base_y = np.full(num_agents, steady_y)
            base_w = np.full(num_agents, steady_w)

            for obj_name, policy in policies.items():
                device = next(policy.parameters()).device
                output_type = policy_output_types.get(
                    obj_name, PolicyOutputType.C_SHARE
                )
                c_vals = []
                for w_val in w_grid:
                    y_dist = base_y.copy()
                    w_dist_raw = base_w.copy()
                    y_dist[0] = steady_y
                    w_dist_raw[0] = w_val
                    w_dist_norm = normalize_w(w_dist_raw, normalization_spec)
                    y_scaled, w_scaled, z_scaled = scale_inputs_numpy(
                        y_dist, w_dist_norm, z_init, input_scale_spec
                    )

                    y_tensor = torch.from_numpy(y_scaled).float().to(device)
                    w_tensor = torch.from_numpy(w_scaled).float().to(device)
                    z_tensor = torch.full(
                        (num_agents,), z_scaled, dtype=torch.float32, device=device
                    )
                    dist_vec = build_dist_features_numpy(y_scaled, w_scaled)
                    dist_tensor = (
                        torch.from_numpy(dist_vec)
                        .float()
                        .to(device)
                        .unsqueeze(0)
                        .expand(num_agents, -1)
                    )
                    with torch.no_grad():
                        if output_type == PolicyOutputType.C_SHARE:
                            share = policy.forward_phi(
                                y_tensor, w_tensor, z_tensor, dist_tensor
                            )
                            c_all = reconstruct_consumption_level(
                                share.cpu().numpy(),
                                w_dist_raw,
                                output_type,
                                normalization_spec
                            )
                            c_vals.append(float(c_all[0]))
                        else:
                            c_all = policy.forward_policy(
                                y_tensor, w_tensor, z_tensor, dist_tensor
                            )
                            c_vals.append(float(c_all[0].item()))
                ax.plot(
                    w_grid, c_vals,
                    color=colors.get(obj_name, 'k'),
                    linestyle=styles.get(obj_name, '-'),
                    label=obj_name
                )
        else:
            for obj_name, sim in simulations.items():
                w_agent = sim['w_path'][:500, 0]
                c_agent = sim['c_path'][:500, 0]
                ax.scatter(w_agent, c_agent, alpha=0.2, s=3,
                          color=colors.get(obj_name, 'k'),
                          label=obj_name)
        
        ax.set_xlabel('Wealth w', fontsize=12)
        ax.set_ylabel('Consumption c', fontsize=12)
        ax.set_title(
            'Panel 1: Consumption Rule\n(Representative Agent)',
            fontsize=12
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Individual wealth simulation
        ax = axes[1]
        
        for obj_name, sim in simulations.items():
            w_agent = sim['w_path'][:min(500, len(sim['w_path'])), 0]
            ax.plot(
                w_agent,
                label=obj_name,
                linewidth=2.5,
                color=colors.get(obj_name, 'k'),
                linestyle=styles.get(obj_name, '-')
            )
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Capital', fontsize=12)
        ax.set_title(
            'Panel 2: Individual Capital\n(Same Agent, Common Shocks)',
            fontsize=12
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Aggregate capital
        ax = axes[2]
        
        for obj_name, sim in simulations.items():
            K_path = sim['K_path'][:min(500, len(sim['K_path']))]
            ax.plot(
                K_path,
                label=obj_name,
                linewidth=2.5,
                color=colors.get(obj_name, 'k'),
                linestyle=styles.get(obj_name, '-')
            )
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Aggregate Capital', fontsize=12)
        ax.set_title(
            'Panel 3: Aggregate Capital\n(All Objectives)',
            fontsize=12
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'comparison_across_objectives.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved: {output_path}")

    def _compute_consumption_rule(
        self,
        policy,
        model,
        simulation_dict: Dict,
        objective_name: str,
        policy_output_type: str,
        normalization_spec: NormalizationSpec,
        input_scale_spec: InputScaleSpec,
        w_plot_max: Optional[float] = None
    ) -> Dict:
        """Compute consumption rule slice with steady-state fixing."""
        import torch

        w_path = simulation_dict['w_path']
        y_path = simulation_dict['y_path']
        z_path = simulation_dict.get('z_path')

        num_agents = w_path.shape[1]
        y_std = model.params.sigma_y / np.sqrt(1 - model.params.rho_y**2)
        y_grid_std = np.linspace(-2.0, 2.0, 7)
        y_levels = y_grid_std * y_std
        if input_scale_spec.enabled:
            w_min = float(input_scale_spec.w_min)
            w_max = float(input_scale_spec.w_max)
            w_grid_source = 'input_scale_range'
        else:
            w_min = 0.0
            w_max = float(np.max(w_path))
            w_grid_source = 'simulation_max'

        if w_plot_max is not None:
            w_max = min(w_max, float(w_plot_max))
            w_grid_source = f"{w_grid_source}_capped"

        w_grid_raw = np.linspace(w_min, w_max, 200)
        steady_y = 0.0
        steady_w = (
            float(input_scale_spec.w_steady)
            if input_scale_spec.enabled
            else float(np.mean(w_path[-1]))
        )
        z_level = 0.0 if z_path is not None else 0.0

        base_y = np.full(num_agents, steady_y)
        base_w_raw = np.full(num_agents, steady_w)

        device = next(policy.parameters()).device
        policy.eval()
        c_grid_raw = np.zeros((len(y_levels), len(w_grid_raw)))
        c_share_grid = np.zeros_like(c_grid_raw)

        with torch.no_grad():
            for idx, y_level in enumerate(y_levels):
                for jdx, w_val_raw in enumerate(w_grid_raw):
                    y_dist = base_y.copy()
                    w_dist_raw = base_w_raw.copy()
                    y_dist[0] = y_level
                    w_dist_raw[0] = w_val_raw

                    w_dist_norm = normalize_w(w_dist_raw, normalization_spec)
                    y_scaled, w_scaled, z_scaled = scale_inputs_numpy(
                        y_dist, w_dist_norm, z_level, input_scale_spec
                    )
                    y_tensor = torch.from_numpy(y_scaled).float().to(device)
                    w_tensor = torch.from_numpy(w_scaled).float().to(device)
                    z_tensor = torch.full(
                        (num_agents,), z_scaled, dtype=torch.float32, device=device
                    )
                    dist_vec = build_dist_features_numpy(y_scaled, w_scaled)
                    dist_tensor = (
                        torch.from_numpy(dist_vec)
                        .float()
                        .to(device)
                        .unsqueeze(0)
                        .expand(num_agents, -1)
                    )

                    if policy_output_type == PolicyOutputType.C_SHARE:
                        share = policy.forward_phi(
                            y_tensor, w_tensor, z_tensor, dist_tensor
                        )
                        c_raw = reconstruct_consumption_level(
                            share.cpu().numpy(),
                            w_dist_raw,
                            policy_output_type,
                            normalization_spec
                        )
                        c_share_grid[idx, jdx] = float(share[0].item())
                        c_grid_raw[idx, jdx] = float(c_raw[0])
                    else:
                        c_all = policy.forward_policy(
                            y_tensor, w_tensor, z_tensor, dist_tensor
                        )
                        c_raw = reconstruct_consumption_level(
                            c_all.cpu().numpy(),
                            w_dist_raw,
                            policy_output_type,
                            normalization_spec
                        )
                        c_grid_raw[idx, jdx] = float(c_raw[0])

        y_labels = [f"y={val:.1f}σ" for val in y_grid_std]

        return {
            'w_grid_raw': w_grid_raw,
            'y_grid_raw': y_levels,
            'y_grid_std': y_grid_std,
            'y_labels': y_labels,
            'c_grid_raw': c_grid_raw,
            'c_share_grid': c_share_grid,
            'w_grid_source': w_grid_source,
            'steady_state': {
                'steady_y': steady_y,
                'steady_w': steady_w,
                'steady_z': z_level
            }
        }

    def _compute_mismatch_checks(
        self,
        objective_name: str,
        policy_plot: Dict,
        simulation_dict: Dict,
        mismatch_config: Dict
    ) -> List[Dict]:
        """Compute simple mismatch diagnostics for policy and wealth scale."""
        c_grid = policy_plot['c_grid_raw']
        share_grid = policy_plot.get('c_share_grid', None)
        w_path = simulation_dict['w_path']

        curvature_threshold = float(mismatch_config.get('curvature_threshold', 1e-4))
        overlap_threshold = float(mismatch_config.get('overlap_threshold', 1e-2))
        share_threshold = float(mismatch_config.get('share_variation_threshold', 1e-4))
        wealth_range_threshold = float(mismatch_config.get('wealth_range_threshold', 2.0))
        wealth_bounds = mismatch_config.get('wealth_quantile_bounds', [0.5, 2.5])
        wealth_lower = float(wealth_bounds[0])
        wealth_upper = float(wealth_bounds[1])

        per_y_curvature = []
        for row in c_grid:
            second_diff = np.diff(row, n=2)
            per_y_curvature.append(float(np.mean(np.abs(second_diff))))
        curvature_metric = float(np.mean(per_y_curvature))

        max_delta = float(np.max(np.max(c_grid, axis=0) - np.min(c_grid, axis=0)))
        mean_delta = float(np.mean(np.max(c_grid, axis=0) - np.min(c_grid, axis=0)))

        share_std = None
        share_ptp = None
        if share_grid is not None and share_grid.size > 0:
            share_std = float(np.std(share_grid))
            share_ptp = float(np.ptp(share_grid))

        w_flat = w_path.flatten()
        p5, p50, p95 = np.percentile(w_flat, [5, 50, 95])
        compression_by_bounds = (p5 > wealth_lower) and (p95 < wealth_upper)
        compression_by_range = (p95 - p5) < wealth_range_threshold

        return [
            {
                'objective': objective_name,
                'check': 'policy_curvature',
                'metric': curvature_metric,
                'threshold': curvature_threshold,
                'flagged': curvature_metric < curvature_threshold,
                'details': {'per_y': per_y_curvature}
            },
            {
                'objective': objective_name,
                'check': 'policy_overlap',
                'metric': max_delta,
                'threshold': overlap_threshold,
                'flagged': max_delta < overlap_threshold,
                'details': {'mean_delta': mean_delta}
            },
            {
                'objective': objective_name,
                'check': 'share_collapse',
                'metric': share_std if share_std is not None else float('nan'),
                'threshold': share_threshold,
                'flagged': bool(share_std is not None and share_std < share_threshold),
                'details': {'share_ptp': share_ptp}
            },
            {
                'objective': objective_name,
                'check': 'wealth_scale_compression',
                'metric': float(p95 - p5),
                'threshold': wealth_range_threshold,
                'flagged': bool(compression_by_bounds or compression_by_range),
                'details': {
                    'p5': float(p5),
                    'p50': float(p50),
                    'p95': float(p95),
                    'compression_by_bounds': bool(compression_by_bounds),
                    'compression_by_range': bool(compression_by_range),
                    'bounds': [wealth_lower, wealth_upper]
                }
            }
        ]

    def _compute_value_function(
        self,
        policy,
        model,
        simulation_dict: Dict,
        normalization_spec: NormalizationSpec,
        input_scale_spec: InputScaleSpec,
        w_plot_max: Optional[float] = None
    ) -> Dict:
        """Compute value function slice with steady-state fixing."""
        import torch

        w_path = simulation_dict['w_path']
        y_path = simulation_dict['y_path']
        z_path = simulation_dict.get('z_path')

        num_agents = w_path.shape[1]
        y_std = model.params.sigma_y / np.sqrt(1 - model.params.rho_y**2)
        y_grid_std = np.linspace(-2.0, 2.0, 7)
        y_levels = y_grid_std * y_std
        if input_scale_spec.enabled:
            w_min = float(input_scale_spec.w_min)
            w_max = float(input_scale_spec.w_max)
        else:
            w_min = 0.0
            w_max = float(np.max(w_path))
        if w_plot_max is not None:
            w_max = min(w_max, float(w_plot_max))
        w_grid_raw = np.linspace(w_min, w_max, 200)

        steady_y = 0.0
        steady_w = (
            float(input_scale_spec.w_steady)
            if input_scale_spec.enabled
            else float(np.mean(w_path[-1]))
        )
        z_level = 0.0 if z_path is not None else 0.0

        base_y = np.full(num_agents, steady_y)
        base_w_raw = np.full(num_agents, steady_w)

        device = next(policy.parameters()).device
        policy.eval()
        v_grid_raw = np.zeros((len(y_levels), len(w_grid_raw)))

        with torch.no_grad():
            for idx, y_level in enumerate(y_levels):
                for jdx, w_val_raw in enumerate(w_grid_raw):
                    y_dist = base_y.copy()
                    w_dist_raw = base_w_raw.copy()
                    y_dist[0] = y_level
                    w_dist_raw[0] = w_val_raw

                    w_dist_norm = normalize_w(w_dist_raw, normalization_spec)
                    y_scaled, w_scaled, z_scaled = scale_inputs_numpy(
                        y_dist, w_dist_norm, z_level, input_scale_spec
                    )
                    y_tensor = torch.from_numpy(y_scaled).float().to(device)
                    w_tensor = torch.from_numpy(w_scaled).float().to(device)
                    z_tensor = torch.full(
                        (num_agents,), z_scaled, dtype=torch.float32, device=device
                    )
                    dist_vec = build_dist_features_numpy(y_scaled, w_scaled)
                    dist_tensor = (
                        torch.from_numpy(dist_vec)
                        .float()
                        .to(device)
                        .unsqueeze(0)
                        .expand(num_agents, -1)
                    )

                    v_all = policy.forward_v(
                        y_tensor, w_tensor, z_tensor, dist_tensor
                    )
                    v_grid_raw[idx, jdx] = float(v_all[0].item())

        y_labels = [f"y={val:.1f}σ" for val in y_grid_std]

        return {
            'w_grid_raw': w_grid_raw,
            'y_grid_raw': y_levels,
            'y_grid_std': y_grid_std,
            'y_labels': y_labels,
            'v_grid_raw': v_grid_raw
        }


def plot_statistics_table(
    stats_df: pd.DataFrame,
    output_dir: Path
):
    """
    Create visualization of statistics table.
    
    Args:
        stats_df: statistics dataframe
        output_dir: output directory
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = stats_df.values
    col_labels = stats_df.columns
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                    cellLoc='center', loc='center',
                    colWidths=[0.1] * len(col_labels))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.title('Summary Statistics by Objective and Agent Count',
             fontsize=14, pad=20)
    plt.tight_layout()
    
    output_path = output_dir / 'statistics_table.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Statistics table saved: {output_path}")

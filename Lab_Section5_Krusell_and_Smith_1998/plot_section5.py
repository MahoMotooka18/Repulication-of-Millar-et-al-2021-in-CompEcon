"""
Plot Generation for Section 5 (Krusell-Smith)

Generates publication-quality plots for training and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict


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
        """Compute a simple moving average."""
        if window <= 1 or len(values) < window:
            return values
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode='valid')

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
        show_raw: bool = True
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
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        # Panel 1: Training loss
        ax = axes[0]
        if 'epoch' in metrics_df.columns and 'objective_train' in metrics_df.columns:
            epochs = metrics_df['epoch'].values
            losses = metrics_df['objective_train'].values
            if show_raw:
                ax.plot(epochs, losses, 'b-', linewidth=1, alpha=0.3)
            smoothed = self._moving_average(losses, smoothing_window)
            if len(smoothed) != len(losses):
                epochs = epochs[-len(smoothed):]
            ax.plot(epochs, smoothed, 'b-', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Training Loss', fontsize=12)
            ax.set_title('Panel 1: Training Loss', fontsize=12)
            if use_log_x:
                ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Consumption policy (c vs w for different y levels)
        ax = axes[1]
        w_path = simulation_dict['w_path']
        c_path = simulation_dict['c_path']
        y_path = simulation_dict['y_path']

        if policy is not None and model is not None:
            num_agents = w_path.shape[1]
            y_std = model.params.sigma_y / np.sqrt(1 - model.params.rho_y**2)
            y_levels = np.linspace(-2 * y_std, 2 * y_std, 7)
            w_grid = np.linspace(0.0, np.max(w_path), 200)
            colors = plt.cm.viridis(np.linspace(0, 1, len(y_levels)))

            steady_y = float(np.mean(y_path[-1]))
            steady_w = float(np.mean(w_path[-1]))
            base_y = np.full(num_agents, steady_y)
            base_w = np.full(num_agents, steady_w)
            z_level = 0.0

            import torch
            policy.eval()
            with torch.no_grad():
                for idx, y_level in enumerate(y_levels):
                    c_vals = []
                    for w_val in w_grid:
                        y_dist = base_y.copy()
                        w_dist = base_w.copy()
                        y_dist[0] = y_level
                        w_dist[0] = w_val

                        y_tensor = torch.from_numpy(y_dist).float()
                        w_tensor = torch.from_numpy(w_dist).float()
                        z_tensor = torch.full((num_agents,), z_level, dtype=torch.float32)

                        dist_vec = np.concatenate([y_dist, w_dist], axis=0)
                        dist_tensor = torch.from_numpy(dist_vec).float().unsqueeze(0).expand(num_agents, -1)

                        c_all = policy.forward_policy(y_tensor, w_tensor, z_tensor, dist_tensor)
                        c_vals.append(float(c_all[0].item()))
                    ax.plot(w_grid, c_vals, color=colors[idx], label=f'y={y_level:.2f}')
        else:
            y_levels = np.percentile(y_path, [10, 30, 50, 70, 90])
            colors = plt.cm.viridis(np.linspace(0, 1, len(y_levels)))

            for idx, y_level in enumerate(y_levels):
                mask = np.abs(y_path[:, 0] - y_level) < 0.5
                if np.any(mask):
                    w_slice = w_path[mask, 0]
                    c_slice = c_path[mask, 0]
                    ax.scatter(w_slice, c_slice, alpha=0.3, s=5,
                              color=colors[idx],
                              label=f'y={y_level:.1f}')

        ax.plot([0, w_path.max()], [0, w_path.max()],
               'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Wealth w', fontsize=12)
        ax.set_ylabel('Consumption c', fontsize=12)
        ax.set_title('Panel 2: Consumption Policy', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Wealth simulation (random agents)
        ax = axes[2]
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
        ax.set_title('Panel 3: Individual Wealth Simulation', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / f"{objective_name}_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {output_path}")

    def plot_comparison_across_objectives(
        self,
        simulations: Dict[str, Dict],
        output_dir: Path,
        policies: Dict[str, object] = None,
        model=None,
        base_state: Dict = None
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
        
        # Panel 1: Consumption rule for representative agent
        ax = axes[0]
        
        if policies is not None and model is not None and base_state is not None:
            import torch

            w_init = base_state['w_init']
            y_init = base_state['y_init']
            z_init = base_state['z_init']
            num_agents = len(w_init)
            steady_y = float(np.mean(y_init))
            steady_w = float(np.mean(w_init))
            w_grid = np.linspace(0.0, np.max(w_init), 200)

            base_y = np.full(num_agents, steady_y)
            base_w = np.full(num_agents, steady_w)

            for obj_name, policy in policies.items():
                device = next(policy.parameters()).device
                c_vals = []
                for w_val in w_grid:
                    y_dist = base_y.copy()
                    w_dist = base_w.copy()
                    y_dist[0] = steady_y
                    w_dist[0] = w_val

                    y_tensor = torch.from_numpy(y_dist).float().to(device)
                    w_tensor = torch.from_numpy(w_dist).float().to(device)
                    z_tensor = torch.full(
                        (num_agents,), z_init, dtype=torch.float32, device=device
                    )
                    dist_vec = np.concatenate([y_dist, w_dist], axis=0)
                    dist_tensor = (
                        torch.from_numpy(dist_vec)
                        .float()
                        .to(device)
                        .unsqueeze(0)
                        .expand(num_agents, -1)
                    )
                    with torch.no_grad():
                        c_all = policy.forward_policy(
                            y_tensor, w_tensor, z_tensor, dist_tensor
                        )
                    c_vals.append(float(c_all[0].item()))
                ax.plot(
                    w_grid, c_vals,
                    color=colors.get(obj_name, 'k'),
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
            ax.plot(w_agent, label=obj_name, linewidth=2.5,
                   color=colors.get(obj_name, 'k'))
        
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
            ax.plot(K_path, label=obj_name, linewidth=2.5,
                   color=colors.get(obj_name, 'k'))
        
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

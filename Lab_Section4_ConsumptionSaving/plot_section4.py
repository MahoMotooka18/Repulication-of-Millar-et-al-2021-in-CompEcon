"""
Plot Generation for Section 4 Results

Generates 3-panel plots showing:
- Panel A: Training objective (varies by method)
- Panel B: Test Euler residuals
- Panel C: Test lifetime reward
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


class SectionPlotter:
    """Generates publication-quality plots for Section 4."""

    def __init__(self, figsize: tuple = (15, 5)):
        """
        Initialize plotter.

        Args:
            figsize: figure size (width, height)
        """
        self.figsize = figsize
        self.colors = {
            8: '#1f77b4',
            16: '#ff7f0e',
            32: '#2ca02c',
            64: '#d62728'
        }
        self.labels = {
            8: '8x8_relu',
            16: '16x16_relu',
            32: '32x32_relu',
            64: '64x64_relu'
        }

    def _moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        """Compute a simple moving average."""
        if window <= 1 or len(values) < window:
            return values
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode='valid')

    def plot_training_curves(
        self,
        data_by_size: Dict[int, pd.DataFrame],
        objective_name: str,
        output_path: Path,
        use_log_x: bool = True,
        smoothing_window: int = 1,
        show_raw: bool = True
    ):
        """
        Create 3-panel training plot.

        Args:
            data_by_size: dict mapping network size -> DataFrame with
                         columns: epoch, objective_train, test_euler_residual,
                         test_lifetime_reward
            objective_name: 'lifetime_reward', 'euler', or 'bellman'
            output_path: Path to save figure
            use_log_x: use log scale on x-axis
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        # Panel A: Training objective
        ax_a = axes[0]
        for size in sorted(data_by_size.keys()):
            df = data_by_size[size]
            epochs = df['epoch'].to_numpy()
            values = df['objective_train'].to_numpy()
            if show_raw:
                ax_a.plot(
                    epochs,
                    values,
                    color=self.colors[size],
                    linewidth=1,
                    alpha=0.3
                )
            smoothed = self._moving_average(values, smoothing_window)
            if len(smoothed) != len(values):
                epochs = epochs[-len(smoothed):]
            ax_a.plot(
                epochs,
                smoothed,
                label=self.labels[size],
                color=self.colors[size],
                linewidth=2
            )
        ax_a.set_xlabel('Epoch', fontsize=12)
        ax_a.set_ylabel('Training Objective', fontsize=12)
        ax_a.set_title(f'Panel A: Training Loss ({objective_name})',
                       fontsize=12)
        ax_a.legend()
        ax_a.grid(True, alpha=0.3)
        if use_log_x:
            ax_a.set_xscale('log')

        # Panel B: Test Euler residuals
        ax_b = axes[1]
        for size in sorted(data_by_size.keys()):
            df = data_by_size[size]
            epochs = df['epoch'].to_numpy()
            values = df['test_euler_residual_mean'].to_numpy()
            if show_raw:
                ax_b.plot(
                    epochs,
                    values,
                    color=self.colors[size],
                    linewidth=1,
                    alpha=0.3
                )
            smoothed = self._moving_average(values, smoothing_window)
            if len(smoothed) != len(values):
                epochs = epochs[-len(smoothed):]
            ax_b.plot(
                epochs,
                smoothed,
                label=self.labels[size],
                color=self.colors[size],
                linewidth=2
            )
        ax_b.set_xlabel('Epoch', fontsize=12)
        ax_b.set_ylabel('Test Euler Residual (mean)', fontsize=12)
        ax_b.set_title('Panel B: Test Euler Residuals', fontsize=12)
        ax_b.legend()
        ax_b.grid(True, alpha=0.3)
        if use_log_x:
            ax_b.set_xscale('log')

        # Panel C: Test lifetime reward
        ax_c = axes[2]
        for size in sorted(data_by_size.keys()):
            df = data_by_size[size]
            epochs = df['epoch'].to_numpy()
            values = df['test_lifetime_reward_mean'].to_numpy()
            if show_raw:
                ax_c.plot(
                    epochs,
                    values,
                    color=self.colors[size],
                    linewidth=1,
                    alpha=0.3
                )
            smoothed = self._moving_average(values, smoothing_window)
            if len(smoothed) != len(values):
                epochs = epochs[-len(smoothed):]
            ax_c.plot(
                epochs,
                smoothed,
                label=self.labels[size],
                color=self.colors[size],
                linewidth=2
            )
        ax_c.set_xlabel('Epoch', fontsize=12)
        ax_c.set_ylabel('Test Lifetime Reward (mean)', fontsize=12)
        ax_c.set_title('Panel C: Test Lifetime Reward', fontsize=12)
        ax_c.legend()
        ax_c.grid(True, alpha=0.3)
        if use_log_x:
            ax_c.set_xscale('log')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_single_metric_curve(
        self,
        df: pd.DataFrame,
        metric_key: str,
        title: str,
        ylabel: str,
        output_path: Path,
        use_log_x: bool = True,
        smoothing_window: int = 1,
        show_raw: bool = True
    ):
        """Plot a single metric curve for one network size."""
        fig, ax = plt.subplots(figsize=(7, 5))
        epochs = df['epoch'].to_numpy()
        values = df[metric_key].to_numpy()
        if show_raw:
            ax.plot(epochs, values, color='#1f77b4', linewidth=1, alpha=0.3)
        smoothed = self._moving_average(values, smoothing_window)
        if len(smoothed) != len(values):
            epochs = epochs[-len(smoothed):]
        ax.plot(epochs, smoothed, color='#1f77b4', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        if use_log_x:
            ax.set_xscale('log')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_euler_residual_distribution(
        self,
        residuals_by_size: Dict[int, np.ndarray],
        objective_name: str,
        output_path: Path
    ):
        """
        Plot distribution of Euler residuals.

        Args:
            residuals_by_size: dict mapping network size -> residuals array
            objective_name: for title
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        positions = []
        data = []
        labels = []

        for i, size in enumerate(sorted(residuals_by_size.keys())):
            residuals = np.abs(residuals_by_size[size]).flatten()
            positions.append(i)
            data.append(residuals)
            labels.append(self.labels[size])

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            labels=labels
        )

        for patch, size in zip(bp['boxes'], sorted(residuals_by_size.keys())):
            patch.set_facecolor(self.colors[size])
            patch.set_alpha(0.7)

        ax.set_ylabel('Absolute Euler Residual', fontsize=12)
        ax.set_title(f'Euler Residual Distribution ({objective_name})',
                     fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_policy_slice(
        self,
        policy,
        y_fixed: float,
        w_range: np.ndarray,
        objective_name: str,
        output_path: Path,
        device: str = 'cpu'
    ):
        """
        Plot consumption policy as function of wealth (for fixed income).

        Args:
            policy: NeuralNetworkPolicy
            y_fixed: fixed log-income value
            w_range: array of wealth values
            objective_name: for title
            output_path: Path to save figure
            device: 'cpu' or 'cuda'
        """
        import torch

        fig, ax = plt.subplots(figsize=(10, 6))

        y_tensor = torch.full((len(w_range),), y_fixed, dtype=torch.float32,
                              device=device)
        w_tensor = torch.from_numpy(w_range).float().to(device)

        with torch.no_grad():
            c = policy.forward_policy(y_tensor, w_tensor)
            c_np = c.cpu().numpy()
            phi = policy.forward_phi(y_tensor, w_tensor)
            phi_np = phi.cpu().numpy()

        ax.plot(w_range, c_np, 'b-', linewidth=2, label='Consumption c')
        ax.plot(w_range, w_range, 'k--', linewidth=1, label='45-degree line')
        ax_twin = ax.twinx()
        ax_twin.plot(w_range, phi_np, 'r-', linewidth=2,
                     label='Consumption share phi')

        ax.set_xlabel('Wealth w', fontsize=12)
        ax.set_ylabel('Consumption c', fontsize=12, color='b')
        ax_twin.set_ylabel('Consumption share c/w', fontsize=12, color='r')
        ax.set_title(f'Policy: c vs w at y={y_fixed:.2f} ({objective_name})',
                     fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor='b')
        ax_twin.tick_params(axis='y', labelcolor='r')
        ax.set_ylim(bottom=0)
        ax_twin.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

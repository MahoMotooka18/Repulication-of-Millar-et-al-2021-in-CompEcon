"""
Main Training Experiment Runner for Section 4

Orchestrates the entire training pipeline:
- Initialize model and policy
- Training loop with ADAM optimizer
- Periodic evaluation
- Results logging and plotting
"""

import argparse
import yaml
import ast
import math
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import json
from typing import Dict, List, Optional

from model_consumption_saving import (
    ConsumptionSavingModel,
    ConsumptionSavingParams
)
from nn_policy import NeuralNetworkPolicy, PolicyFactory
from objectives import ObjectiveComputer
from evaluator import Evaluator
from plot_section4 import SectionPlotter


class ExperimentRunner:
    """Main experiment runner."""

    def __init__(self, config_path: str, device: str = 'cpu'):
        """
        Initialize experiment runner.

        Args:
            config_path: path to config YAML file
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.initialize_model()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return self._coerce_numeric(config)

    def _coerce_numeric(self, obj):
        """Safely evaluate numeric expressions in config (e.g., '0.2 * sqrt(...)')."""
        if isinstance(obj, dict):
            return {k: self._coerce_numeric(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._coerce_numeric(v) for v in obj]
        if isinstance(obj, str):
            try:
                return self._safe_eval(obj)
            except Exception:
                return obj
        return obj

    def _safe_eval(self, expr: str):
        """Evaluate simple numeric expressions with math functions only."""
        allowed_names = {
            'sqrt': math.sqrt,
            'exp': math.exp,
            'log': math.log,
            'pi': math.pi,
            'e': math.e
        }

        node = ast.parse(expr, mode='eval')

        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name):
                if subnode.id not in allowed_names:
                    raise ValueError("Unsafe name")
            elif isinstance(subnode, ast.Call):
                if not isinstance(subnode.func, ast.Name):
                    raise ValueError("Unsafe call")
                if subnode.func.id not in allowed_names:
                    raise ValueError("Unsafe call")
            elif isinstance(subnode, (ast.BinOp, ast.UnaryOp, ast.Expression,
                                      ast.Load, ast.Constant, ast.Add, ast.Sub,
                                      ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)):
                continue
            else:
                raise ValueError("Unsafe expression")

        return eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}, allowed_names)

    def setup_directories(self):
        """Create output directories."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"{timestamp}"
        self.output_dir = Path(self.config.get(
            'output_dir',
            'outputs/section4'
        )) / run_id

        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.plot_dir = self.output_dir / 'plots'

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {self.output_dir}")

    def initialize_model(self):
        """Initialize consumption-saving model."""
        params = ConsumptionSavingParams(
            gamma=self.config['model']['gamma'],
            beta=self.config['model']['beta'],
            r=self.config['model']['r'],
            rho=self.config['model']['rho'],
            sigma=self.config['model']['sigma'],
            T=self.config['model']['horizon']
        )
        self.model = ConsumptionSavingModel(params)
        self.objective_computer = ObjectiveComputer(self.model, self.device)
        self.evaluator = Evaluator(self.model, self.device)

    def run_experiment(self):
        """Run the full experiment."""
        objective_names = self.config['training']['objective']
        network_sizes = self.config['training']['network_sizes']
        num_epochs = self.config['training']['num_epochs']
        learning_rate = self.config['training']['learning_rate']
        batch_size = self.config['training']['batch_size']

        all_metrics = {}
        all_residuals = {}
        all_policies = {}

        for objective_name in objective_names:
            all_metrics[objective_name] = {}
            all_residuals[objective_name] = {}
            all_policies[objective_name] = {}
            for net_size in network_sizes:
                print(f"\n{'='*60}")
                print(f"Training objective: {objective_name}")
                print(f"Network size: {net_size}x{net_size}")
                print(f"{'='*60}")

                # Create policy
                policy = PolicyFactory.create_policy(
                    hidden_size=net_size,
                    device=self.device
                ).train()

                # Create optimizer
                optimizer = optim.Adam(
                    policy.parameters(),
                    lr=learning_rate
                )

                # Training loop
                metrics = self.training_loop(
                    policy,
                    optimizer,
                    objective_name,
                    num_epochs,
                    batch_size,
                    net_size
                )

                all_metrics[objective_name][net_size] = metrics
                all_policies[objective_name][net_size] = policy

                # Save checkpoint
                self._save_checkpoint(
                    policy,
                    objective_name,
                    net_size
                )

            # Generate diagnostics for plots
            for net_size, policy in all_policies[objective_name].items():
                residuals = self._collect_residuals(policy)
                all_residuals[objective_name][net_size] = residuals

            # Generate plots and save results
            self._save_metrics(all_metrics[objective_name], objective_name)
            self._generate_plots(
                all_metrics[objective_name],
                all_residuals[objective_name],
                all_policies[objective_name],
                objective_name
            )
        self._save_metrics_all(all_metrics)
        self._save_config()

    def training_loop(
        self,
        policy: NeuralNetworkPolicy,
        optimizer,
        objective_name: str,
        num_epochs: int,
        batch_size: int,
        net_size: int
    ) -> pd.DataFrame:
        """
        Run training loop.

        Args:
            policy: neural network policy
            optimizer: PyTorch optimizer
            objective_name: 'lifetime_reward', 'euler', or 'bellman'
            num_epochs: number of training epochs
            batch_size: batch size
            net_size: network size (for logging)

        Returns:
            metrics DataFrame
        """
        metrics = []
        rng = np.random.default_rng(self.config['seed'])
        w_min, w_max = self.config['training']['wealth_range']

        eval_interval = self.config['training'].get(
            'eval_interval',
            max(1, num_epochs // 50)
        )
        debug_cfg = self.config.get('debug', {})
        debug_enabled = bool(debug_cfg.get('enabled', False))
        debug_interval = int(debug_cfg.get('interval', eval_interval))

        start_time = time.time()

        for epoch in range(num_epochs):
            # Generate batch
            ergodic_std = self._ergodic_income_std()
            y_batch = torch.from_numpy(
                rng.normal(0.0, ergodic_std, batch_size)
            ).float().to(self.device)
            w_batch = torch.from_numpy(
                rng.uniform(w_min, w_max, batch_size)
            ).float().to(self.device)

            # Training step
            optimizer.zero_grad()

            if objective_name == 'lifetime_reward':
                loss = self._train_step_lifetime_reward(
                    policy, y_batch, w_batch, rng
                )
            elif objective_name == 'euler':
                if debug_enabled and (epoch % debug_interval == 0):
                    self._log_aioshocks(rng, batch_size, tag=f"epoch={epoch}")
                loss = self._train_step_euler(
                    policy, y_batch, w_batch, rng
                )
            elif objective_name == 'bellman':
                if debug_enabled and (epoch % debug_interval == 0):
                    self._log_aioshocks(rng, batch_size, tag=f"epoch={epoch}")
                loss = self._train_step_bellman(
                    policy, y_batch, w_batch, rng
                )
            else:
                raise ValueError(f"Unknown objective: {objective_name}")

            loss.backward()
            optimizer.step()

            # Evaluation
            if (epoch + 1) % eval_interval == 0 or epoch == 0:
                wall_time = time.time() - start_time
                eval_metrics = self._evaluate_step(
                    policy,
                    epoch,
                    loss.item(),
                    wall_time,
                    objective_name,
                    net_size
                )
                metrics.append(eval_metrics)
                if debug_enabled and (epoch % debug_interval == 0):
                    self._log_constraints(
                        f"epoch={epoch}",
                        policy,
                        rng
                    )

                if epoch % (eval_interval * 10) == 0:
                    print(
                        f"Epoch {epoch}/{num_epochs} | "
                        f"Loss: {loss.item():.6e} | "
                        f"Euler FB (mean): "
                        f"{eval_metrics['test_euler_residual_mean']:.6e}"
                    )

        return pd.DataFrame(metrics)

    def _log_constraints(self, tag: str, policy, rng):
        """Log constraint diagnostics on a small batch."""
        n = 512
        ergodic_std = self._ergodic_income_std()
        y_batch = torch.from_numpy(
            rng.normal(0.0, ergodic_std, n)
        ).float().to(self.device)
        w_min, w_max = self.config['training']['wealth_range']
        w_batch = torch.from_numpy(
            rng.uniform(w_min, w_max, n)
        ).float().to(self.device)

        with torch.no_grad():
            c = w_batch * policy.forward_phi(y_batch, w_batch)
            c = torch.clamp(c, min=torch.zeros_like(w_batch), max=w_batch)
            c_np = c.cpu().numpy()
            w_np = w_batch.cpu().numpy()
            w_next = (
                self.model.params.r * (w_np - c_np) +
                np.exp(y_batch.cpu().numpy())
            )

        c_lt_0 = np.mean(c_np < 0)
        c_gt_w = np.mean(c_np > w_np)
        w_lt_0 = np.mean(w_np < 0)
        msg = (f"[debug] {tag} min(c)={np.min(c_np):.4g} max(c)={np.max(c_np):.4g} "
               f"min(w)={np.min(w_np):.4g} max(w)={np.max(w_np):.4g} "
               f"min(w_next)={np.min(w_next):.4g} "
               f"share(c<0)={c_lt_0:.3g} share(c>w)={c_gt_w:.3g} share(w<0)={w_lt_0:.3g}")
        print(msg)

    def _log_aioshocks(self, rng, batch_size: int, tag: str = ""):
        """Log AiO shock independence diagnostics."""
        eps1 = rng.standard_normal(batch_size)
        eps2 = rng.standard_normal(batch_size)
        same = np.allclose(eps1, eps2)
        corr = np.corrcoef(eps1, eps2)[0, 1]
        print(f"[debug] AiO shocks {tag}: same={same} corr={corr:.4g}")

    def _train_step_lifetime_reward(
        self,
        policy,
        y_batch,
        w_batch,
        rng
    ):
        """Training step for lifetime reward objective."""
        T = self.config['model']['horizon']
        y_path = torch.zeros((T, len(y_batch)), device=self.device)
        w_path = torch.zeros((T, len(w_batch)), device=self.device)
        c_path = torch.zeros((T, len(w_batch)), device=self.device)

        y_t = y_batch.clone()
        w_t = w_batch.clone()

        for t in range(T):
            y_path[t] = y_t
            w_path[t] = w_t

            phi = policy.forward_phi(y_t, w_t)
            c_t = w_t * phi
            c_t = torch.clamp(c_t, min=torch.zeros_like(w_t), max=w_t)
            c_path[t] = c_t

            eps_t = torch.from_numpy(
                rng.standard_normal(len(w_t))
            ).float().to(self.device)
            w_t = self._state_transition_tensor(w_t, c_t, y_t)
            y_t = self._income_transition_tensor(y_t, eps_t)

        loss = self.objective_computer.lifetime_reward_objective(
            y_path, w_path, c_path
        )
        return loss

    def _train_step_euler(
        self,
        policy,
        y_batch,
        w_batch,
        rng
    ):
        """Training step for Euler equation objective."""
        eps1 = torch.from_numpy(
            rng.standard_normal(len(y_batch))
        ).float().to(self.device)
        eps2 = torch.from_numpy(
            rng.standard_normal(len(y_batch))
        ).float().to(self.device)

        loss = self.objective_computer.euler_objective(
            policy,
            y_batch,
            w_batch,
            eps1,
            eps2,
            nu_h=self.config['training'].get('nu_h', 1.0),
            nu=self.config['training'].get('nu', 1.0)
        )
        return loss

    def _train_step_bellman(
        self,
        policy,
        y_batch,
        w_batch,
        rng
    ):
        """Training step for Bellman equation objective."""
        eps1 = torch.from_numpy(
            rng.standard_normal(len(y_batch))
        ).float().to(self.device)
        eps2 = torch.from_numpy(
            rng.standard_normal(len(y_batch))
        ).float().to(self.device)

        loss = self.objective_computer.bellman_objective(
            policy,
            y_batch,
            w_batch,
            eps1,
            eps2,
            nu_h=self.config['training'].get('nu_h', 1.0),
            nu=self.config['training'].get('nu', 1.0)
        )
        return loss

    def _income_transition_tensor(
        self,
        y_t: torch.Tensor,
        eps_t: torch.Tensor
    ) -> torch.Tensor:
        """Income transition on tensors."""
        y_next = (self.model.params.rho * y_t +
                  self.model.params.sigma * eps_t)
        return y_next

    def _state_transition_tensor(
        self,
        w_t: torch.Tensor,
        c_t: torch.Tensor,
        y_t: torch.Tensor
    ) -> torch.Tensor:
        """State transition on tensors."""
        w_next = (self.model.params.r * (w_t - c_t) +
                  torch.exp(y_t))
        return w_next

    def _evaluate_step(
        self,
        policy,
        epoch,
        loss_value,
        wall_time,
        objective_name,
        net_size
    ) -> Dict:
        """Evaluate policy and return metrics."""
        policy.eval()

        with torch.no_grad():
            # Generate test data
            n_test = 8192
            rng = np.random.default_rng(self.config['seed'] + 1)

            ergodic_std = self._ergodic_income_std()
            y_test = torch.from_numpy(
                rng.normal(0.0, ergodic_std, n_test)
            ).float().to(self.device)
            w_min, w_max = self.config['training']['wealth_range']
            w_test = torch.from_numpy(
                rng.uniform(w_min, w_max, n_test)
            ).float().to(self.device)

            # Generate shocks
            num_steps = self.config['model']['horizon']
            eps_test = rng.standard_normal((num_steps, n_test))
            eps_test = torch.from_numpy(eps_test).float().to(self.device)

            # Evaluate
            metrics_dict = self.evaluator.evaluate(
                policy,
                y_test,
                w_test,
                eps_test,
                num_steps=num_steps
            )

            stats = self.evaluator.compute_statistics(metrics_dict)

        policy.train()

        result = {
            'epoch': epoch,
            'objective_train': loss_value,
            'test_euler_residual_mean': stats['euler_fb_mean'],
            'test_euler_residual_p50': stats['euler_fb_p50'],
            'test_euler_residual_p90': stats['euler_fb_p90'],
            'test_euler_residual_max': stats['euler_fb_max'],
            'test_lifetime_reward_mean': stats['lifetime_reward_mean'],
            'wall_time_sec': wall_time,
            'seed': self.config['seed'],
            'net_size': net_size,
            'objective_name': objective_name
        }

        return result

    def _ergodic_income_std(self) -> float:
        """Compute ergodic std deviation for AR(1) income."""
        rho = self.model.params.rho
        sigma = self.model.params.sigma
        return sigma / np.sqrt(1.0 - rho**2)

    def _save_checkpoint(
        self,
        policy: NeuralNetworkPolicy,
        objective_name: str,
        net_size: int
    ):
        """Save policy checkpoint."""
        checkpoint_path = (self.checkpoint_dir /
                           f"policy_{objective_name}_{net_size}.pt")
        torch.save(policy.state_dict(), checkpoint_path)

    def _save_metrics(self, metrics_by_size: Dict, objective_name: str):
        """Save metrics to CSV."""
        dfs = []
        for net_size, df in metrics_by_size.items():
            df = df.copy()
            df['net_size'] = net_size
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        csv_path = self.output_dir / f'metrics_{objective_name}.csv'
        combined.to_csv(csv_path, index=False)
        print(f"\nMetrics saved to {csv_path}")

    def _save_metrics_all(self, all_metrics: Dict):
        """Save a combined metrics.csv across objectives and sizes."""
        dfs = []
        for objective_name, metrics_by_size in all_metrics.items():
            for net_size, df in metrics_by_size.items():
                df_copy = df.copy()
                df_copy['net_size'] = net_size
                df_copy['objective_name'] = objective_name
                dfs.append(df_copy)

        if not dfs:
            return

        combined = pd.concat(dfs, ignore_index=True)
        csv_path = self.output_dir / 'metrics.csv'
        combined.to_csv(csv_path, index=False)
        print(f"\nMetrics saved to {csv_path}")

    def _generate_plots(
        self,
        all_metrics: Dict,
        all_residuals: Dict,
        all_policies: Dict,
        objective_name: str
    ):
        """Generate plots."""
        plotter = SectionPlotter()
        plot_cfg = self.config.get('plotting', {})
        smoothing_window = int(plot_cfg.get('smoothing_window', 1))
        show_raw = bool(plot_cfg.get('show_raw', True))

        # Group by network size for plotting
        data_by_size = all_metrics

        plot_path = (self.plot_dir /
                     f"training_curves_{objective_name}.png")
        plotter.plot_training_curves(
            data_by_size,
            objective_name,
            plot_path,
            smoothing_window=smoothing_window,
            show_raw=show_raw
        )
        print(f"Plot saved to {plot_path}")

    def _save_config(self):
        """Save configuration."""
        config_path = self.output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

    def _collect_residuals(self, policy: NeuralNetworkPolicy) -> np.ndarray:
        """Collect Euler residuals for distribution plots."""
        policy.eval()
        with torch.no_grad():
            n_test = 8192
            rng = np.random.default_rng(self.config['seed'] + 1)
            ergodic_std = self._ergodic_income_std()
            y_test = torch.from_numpy(
                rng.normal(0.0, ergodic_std, n_test)
            ).float().to(self.device)
            w_min, w_max = self.config['training']['wealth_range']
            w_test = torch.from_numpy(
                rng.uniform(w_min, w_max, n_test)
            ).float().to(self.device)
            eps_test = rng.standard_normal(
                (self.config['model']['horizon'], n_test)
            )
            eps_test = torch.from_numpy(eps_test).float().to(self.device)

            metrics_dict = self.evaluator.evaluate(
                policy,
                y_test,
                w_test,
                eps_test,
                num_steps=self.config['model']['horizon']
            )
        policy.train()
        return metrics_dict['euler_residual_fb']


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run Section 4 experiment'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/section4.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for training'
    )

    args = parser.parse_args()

    runner = ExperimentRunner(args.config, device=args.device)
    runner.run_experiment()


if __name__ == '__main__':
    main()

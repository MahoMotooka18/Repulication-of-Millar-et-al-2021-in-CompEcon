"""
Complete Training and Evaluation for Section 5 (Krusell-Smith)

Orchestrates training, evaluation, plotting, and table generation
according to section5_math.md requirements (Sections 5-7).
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
from typing import Dict, List

from model_ks1998 import KrusellSmithModel, KrusellSmithParams
from nn_policy_ks import KSPolicyFactory
from objectives_ks import KSObjectiveComputer
from evaluator_ks import KSEvaluator
from plot_section5 import KSPlotter
from report_ks import KSReporter


class KSExperimentRunnerComplete:
    """Complete experiment runner with all outputs per section5_math.md."""

    def __init__(self, config_path: str, device: str = 'cpu'):
        """
        Initialize experiment runner.
        
        Args:
            config_path: path to YAML config
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.initialize_model()
        self.plotter = KSPlotter()
        self.reporter = KSReporter()
        
        self.all_results = []  # Store all results for comparison
        self.all_simulations = {}  # Store simulations by objective
        self.all_policies = {}  # Store policies by objective

    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration."""
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
        """Create output directories per section5_math.md."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = timestamp
        
        self.output_dir = Path(
            self.config.get('output_dir', 'outputs/section5')
        ) / run_id
        
        # Create required directories (Section 5.1-5.4)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.plots_dir = self.output_dir / 'plots'
        self.tables_dir = self.output_dir / 'tables'
        self.comparison_dir = self.output_dir / 'comparison'
        
        for d in [self.checkpoint_dir, self.plots_dir,
                  self.tables_dir, self.comparison_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")

    def initialize_model(self):
        """Initialize KS model."""
        params = KrusellSmithParams(
            gamma=self.config['model']['gamma'],
            beta=self.config['model']['beta'],
            alpha=self.config['model']['alpha'],
            delta=self.config['model']['delta'],
            rho_y=self.config['model']['rho_y'],
            sigma_y=self.config['model']['sigma_y'],
            rho_z=self.config['model']['rho_z'],
            sigma_z=self.config['model']['sigma_z'],
            num_agents=self.config['model']['num_agents'],
            horizon=self.config['model']['horizon']
        )
        self.model = KrusellSmithModel(params)
        self.objective_computer = KSObjectiveComputer(
            self.model, self.device
        )
        self.evaluator = KSEvaluator(self.model, self.device)

    def run_full_experiment(self):
        """Run full experimental grid (Section 5-7 structure)."""
        objectives = self.config['training']['objectives']
        agent_counts = self.config['training']['agent_counts']

        print("\n" + "="*80)
        print("FULL EXPERIMENTAL GRID (SECTION 5-7)")
        print("="*80)
        print(f"Objectives: {objectives}")
        print(f"Agent counts: {agent_counts}")
        print(f"Grid size: {len(objectives)} × {len(agent_counts)}")
        print("="*80 + "\n")

        # Grid loop
        for obj_name in objectives:
            print(f"\n{'='*70}")
            print(f"OBJECTIVE: {obj_name.upper()}")
            print(f"{'='*70}")

            obj_results = []
            obj_simulations = {}
            obj_policies = {}

            for num_agents in agent_counts:
                print(f"\n  Training with {num_agents} agents...")

                # Update model
                self.model.params.num_agents = num_agents

                # Train and evaluate
                result = self.train_and_evaluate(
                    obj_name, num_agents
                )

                if result:
                    obj_results.append(result)
                    obj_simulations[num_agents] = result['simulation']
                    obj_policies[num_agents] = result['policy']
                    self.all_results.append(result)

            # Save objective-specific outputs
            if obj_results:
                self._save_objective_outputs(
                    obj_name, obj_results, obj_simulations
                )

            # Store for cross-objective comparison
            if obj_simulations:
                self.all_simulations[obj_name] = obj_simulations
            if obj_policies:
                self.all_policies[obj_name] = obj_policies

        # Generate cross-objective comparison (Section 5.4, 5.5)
        self._generate_cross_objective_outputs()

        # Final comprehensive tables (Section 5.4, 5.5)
        self._save_comprehensive_tables()

        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE")
        print("="*80)

    def train_and_evaluate(
        self,
        objective_name: str,
        num_agents: int
    ) -> Dict:
        """Train and evaluate for given objective and agent count."""
        # Create policy
        policy = KSPolicyFactory.create_policy(
            hidden_size=self.config['training']['hidden_size'],
            num_agents=num_agents,
            device=self.device
        ).train()

        # Initialize state
        rng = np.random.default_rng(self.config['seed'])
        w_init = rng.exponential(1.0, num_agents)
        y_init = rng.standard_normal(num_agents) * 0.1
        z_init = 0.0
        K_init = np.sum(w_init)

        # Optimizer
        optimizer = optim.Adam(
            policy.parameters(),
            lr=self.config['training']['learning_rate']
        )

        # Training loop
        total_periods = self.config['training'].get(
            'simulation_length',
            self.config['training'].get('num_epochs', 100)
        )
        train_every = self.config['training'].get('train_every', 1)
        train_points = min(
            self.config['training'].get('train_points', 100),
            num_agents
        )
        pretrain_value_iters = (
            self.config['training'].get('pretrain_value_iters', 0)
            if objective_name == 'bellman' else 0
        )
        eval_interval = self.config['training'].get('eval_interval', 100)
        debug_cfg = self.config.get('debug', {})
        debug_enabled = bool(debug_cfg.get('enabled', False))
        debug_interval = int(debug_cfg.get('interval', eval_interval))
        metrics = []
        start_time = time.time()
        update_step = 0

        w_t = w_init.copy()
        y_t = y_init.copy()
        z_t = z_init
        K_t = K_init

        for period in range(total_periods):
            # Simulate one period
            y_t = self.model.normalize_productivity(y_t)
            L_t = self.model.total_labor(y_t)
            R_t, W_t = self.model.factor_prices(z_t, K_t, L_t)

            dist_vec = np.concatenate([y_t, w_t], axis=0)

            y_tensor = torch.from_numpy(y_t).float().to(self.device)
            w_tensor = torch.from_numpy(w_t).float().to(self.device)
            z_tensor = torch.full(
                (num_agents,), z_t,
                dtype=torch.float32,
                device=self.device
            )
            dist_tensor = (
                torch.from_numpy(dist_vec)
                .float().to(self.device)
                .unsqueeze(0).expand(num_agents, -1)
            )

            with torch.no_grad():
                c_t = policy.forward_policy(
                    y_tensor, w_tensor, z_tensor, dist_tensor
                )
                c_t = torch.clamp(
                    c_t, min=torch.zeros_like(w_tensor), max=w_tensor
                )
                c_t_np = c_t.cpu().numpy()

            k_next_full = w_t - c_t_np
            if debug_enabled and (period % debug_interval == 0):
                w_next_dbg = None
                self._log_constraints(
                    f"period={period}",
                    w_t,
                    c_t_np,
                    w_next_dbg
                )

            # Train every N periods
            if period % train_every == 0:
                update_step += 1
                optimizer.zero_grad()

                sample_size = min(train_points, num_agents)
                replace = num_agents < sample_size
                sample_idx = rng.choice(
                    num_agents, size=sample_size, replace=replace
                )

                y_sample = y_tensor[sample_idx]
                w_sample = w_tensor[sample_idx]
                z_sample = z_tensor[sample_idx]
                dist_sample = dist_tensor[sample_idx]

                eps1_y_full = rng.standard_normal(num_agents)
                eps2_y_full = rng.standard_normal(num_agents)
                eps1_z = rng.standard_normal()
                eps2_z = rng.standard_normal()
                if debug_enabled and (update_step % debug_interval == 0):
                    self._log_aioshocks(eps1_y_full, eps2_y_full, eps1_z, eps2_z)

                y_next_full_1 = self.model.income_transition(
                    y_t, eps1_y_full
                )
                y_next_full_1 = self.model.normalize_productivity(
                    y_next_full_1
                )
                y_next_full_2 = self.model.income_transition(
                    y_t, eps2_y_full
                )
                y_next_full_2 = self.model.normalize_productivity(
                    y_next_full_2
                )

                z_next_1 = self.model.aggregate_productivity_transition(
                    z_t, eps1_z
                )
                z_next_2 = self.model.aggregate_productivity_transition(
                    z_t, eps2_z
                )

                K_next = np.sum(k_next_full)
                L_next_1 = self.model.total_labor(y_next_full_1)
                L_next_2 = self.model.total_labor(y_next_full_2)
                R_next_1, W_next_1 = self.model.factor_prices(
                    z_next_1, K_next, L_next_1
                )
                R_next_2, W_next_2 = self.model.factor_prices(
                    z_next_2, K_next, L_next_2
                )

                w_next_full_1 = self.model.state_transition(
                    w_t, c_t_np, y_next_full_1, R_next_1, W_next_1
                )
                w_next_full_2 = self.model.state_transition(
                    w_t, c_t_np, y_next_full_2, R_next_2, W_next_2
                )

                dist_next_1 = np.concatenate(
                    [y_next_full_1, w_next_full_1], axis=0
                )
                dist_next_2 = np.concatenate(
                    [y_next_full_2, w_next_full_2], axis=0
                )

                y_next_1_sample = torch.from_numpy(
                    y_next_full_1[sample_idx]
                ).float().to(self.device)
                y_next_2_sample = torch.from_numpy(
                    y_next_full_2[sample_idx]
                ).float().to(self.device)
                w_next_1_sample = torch.from_numpy(
                    w_next_full_1[sample_idx]
                ).float().to(self.device)
                w_next_2_sample = torch.from_numpy(
                    w_next_full_2[sample_idx]
                ).float().to(self.device)
                z_next_1_sample = torch.full(
                    (sample_size,), z_next_1,
                    dtype=torch.float32,
                    device=self.device
                )
                z_next_2_sample = torch.full(
                    (sample_size,), z_next_2,
                    dtype=torch.float32,
                    device=self.device
                )
                dist_next_1_sample = torch.from_numpy(
                    dist_next_1
                ).float().to(self.device).unsqueeze(0).expand(
                    sample_size, -1
                )
                dist_next_2_sample = torch.from_numpy(
                    dist_next_2
                ).float().to(self.device).unsqueeze(0).expand(
                    sample_size, -1
                )

                if objective_name == 'lifetime_reward':
                    loss = self._train_step_lifetime_reward(
                        policy,
                        y_t,
                        w_t,
                        z_t,
                        sample_idx,
                        rng
                    )
                elif objective_name == 'euler':
                    loss = (
                        self.objective_computer.euler_objective(
                            policy, y_sample, w_sample, z_sample,
                            dist_sample,
                            y_next_1_sample, w_next_1_sample, z_next_1_sample,
                            dist_next_1_sample,
                            y_next_2_sample, w_next_2_sample, z_next_2_sample,
                            dist_next_2_sample,
                            R_next_1, R_next_2,
                            nu_h=self.config['training'].get('nu_h', 1.0)
                        )
                    )
                elif objective_name == 'bellman':
                    self._set_bellman_pretrain(
                        policy, update_step <= pretrain_value_iters
                    )
                    loss = (
                        self.objective_computer.bellman_objective(
                            policy, y_sample, w_sample, z_sample,
                            dist_sample,
                            y_next_1_sample, w_next_1_sample, z_next_1_sample,
                            dist_next_1_sample,
                            y_next_2_sample, w_next_2_sample, z_next_2_sample,
                            dist_next_2_sample,
                            nu_h=self.config['training'].get('nu_h', 1.0),
                            nu=self.config['training'].get('nu', 1.0)
                        )
                    )

                loss.backward()
                optimizer.step()

                if update_step % eval_interval == 0:
                    wall_time = time.time() - start_time
                    simulation_eval = self.evaluator.evaluate_simulation(
                        policy,
                        w_t,
                        y_t,
                        z_t,
                        K_t,
                        T=self.config['training'].get('eval_horizon', 1000),
                        seed=self.config.get('seed', 42) + update_step
                    )
                    stats = self.evaluator.compute_statistics(
                        simulation_eval, burn_in=100
                    )
                    euler_stats = self.evaluator.compute_euler_residuals(
                        simulation_eval, burn_in=100
                    )
                    lr_stats = self.evaluator.compute_lifetime_reward(
                        simulation_eval, burn_in=100
                    )

                    metric = {
                        'epoch': update_step,
                        'objective_train': loss.item(),
                        'test_euler_residual_mean': euler_stats['euler_residual_mean'],
                        'test_euler_residual_p50': euler_stats['euler_residual_p50'],
                        'test_euler_residual_p90': euler_stats['euler_residual_p90'],
                        'test_lifetime_reward_mean': lr_stats['lifetime_reward_mean'],
                        'aggregate_capital_mean': stats['K_mean'],
                        'aggregate_capital_std': stats['K_std'],
                        'wall_time_sec': wall_time,
                        'seed': self.config['seed'],
                        'net_size': self.config['training']['hidden_size'],
                        'objective_name': objective_name,
                        'num_agents': num_agents
                    }
                    metrics.append(metric)

            # Transition
            eps_y = rng.standard_normal(num_agents)
            eps_z = rng.standard_normal()

            y_next = self.model.income_transition(y_t, eps_y)
            y_next = self.model.normalize_productivity(y_next)
            z_next = self.model.aggregate_productivity_transition(z_t, eps_z)
            K_next = np.sum(k_next_full)
            L_next = self.model.total_labor(y_next)
            R_next, W_next = self.model.factor_prices(z_next, K_next, L_next)
            w_next = self.model.state_transition(
                w_t, c_t_np, y_next, R_next, W_next
            )
            if debug_enabled and (period % debug_interval == 0):
                self._log_constraints(
                    f"transition period={period}",
                    w_t,
                    c_t_np,
                    w_next
                )

            w_t = w_next
            y_t = y_next
            z_t = z_next
            K_t = K_next

        # Final evaluation
        policy.eval()
        simulation = self.evaluator.evaluate_simulation(
            policy, w_t, y_t, z_t, K_t,
            T=self.config['training'].get(
                'eval_horizon', 1000
            ),
            seed=self.config.get('seed', 42)
        )
        stats = self.evaluator.compute_statistics(
            simulation, burn_in=100
        )
        euler_stats = self.evaluator.compute_euler_residuals(
            simulation, burn_in=100
        )
        lr_stats = self.evaluator.compute_lifetime_reward(
            simulation, burn_in=100
        )

        total_time = time.time() - start_time
        stats['time_sec'] = float(total_time)

        # Save checkpoint
        self._save_checkpoint(
            policy, objective_name, num_agents
        )

        return {
            'objective': objective_name,
            'num_agents': num_agents,
            'metrics': pd.DataFrame(metrics),
            'statistics': {**stats, **euler_stats, **lr_stats},
            'simulation': simulation,
            'policy': policy
        }

    def _log_constraints(self, tag: str, w: np.ndarray, c: np.ndarray, w_next: np.ndarray):
        """Log constraint diagnostics for a batch."""
        c_lt_0 = np.mean(c < 0)
        c_gt_w = np.mean(c > w)
        w_lt_0 = np.mean(w < 0)
        msg = (f"[debug] {tag} min(c)={np.min(c):.4g} max(c)={np.max(c):.4g} "
               f"min(w)={np.min(w):.4g} max(w)={np.max(w):.4g} "
               f"share(c<0)={c_lt_0:.3g} share(c>w)={c_gt_w:.3g} share(w<0)={w_lt_0:.3g}")
        if w_next is not None:
            msg += f" min(w_next)={np.min(w_next):.4g} max(w_next)={np.max(w_next):.4g}"
        print(msg)

    def _log_aioshocks(self, eps1_y: np.ndarray, eps2_y: np.ndarray, eps1_z: float, eps2_z: float):
        """Log AiO shock independence diagnostics."""
        same_y = np.allclose(eps1_y, eps2_y)
        same_z = np.isclose(eps1_z, eps2_z)
        corr_y = np.corrcoef(eps1_y, eps2_y)[0, 1]
        print(f"[debug] AiO shocks: same_y={same_y} corr_y={corr_y:.4g} same_z={same_z}")

    def _set_bellman_pretrain(self, policy, pretrain: bool):
        """Freeze policy heads during Bellman pretraining."""
        for param in policy.parameters():
            param.requires_grad = not pretrain

        if pretrain:
            for param in policy.v_output.parameters():
                param.requires_grad = True
            policy.v_intercept.requires_grad = True

    def _normalize_productivity_torch(self, y: torch.Tensor) -> torch.Tensor:
        """Normalize productivity so mean exp(y)=1 (torch version)."""
        mean_exp = torch.mean(torch.exp(y))
        return y - torch.log(mean_exp + 1e-12)

    def _train_step_lifetime_reward(
        self,
        policy,
        y_init: np.ndarray,
        w_init: np.ndarray,
        z_init: float,
        sample_idx: np.ndarray,
        rng: np.random.Generator
    ) -> torch.Tensor:
        """Lifetime reward training step using a finite-horizon rollout."""
        num_agents = len(w_init)
        T = int(self.config['model'].get('horizon', self.model.params.horizon))

        y_t = torch.from_numpy(y_init).float().to(self.device)
        w_t = torch.from_numpy(w_init).float().to(self.device)
        z_t = torch.tensor(float(z_init), device=self.device)
        sample_idx_t = torch.from_numpy(sample_idx).long().to(self.device)

        c_path = []
        w_path = []

        alpha = self.model.params.alpha
        delta = self.model.params.delta
        rho_y = self.model.params.rho_y
        sigma_y = self.model.params.sigma_y
        rho_z = self.model.params.rho_z
        sigma_z = self.model.params.sigma_z

        for _ in range(T):
            y_t = self._normalize_productivity_torch(y_t)
            dist_vec = torch.cat([y_t, w_t], dim=0)
            dist_tensor = dist_vec.unsqueeze(0).expand(num_agents, -1)
            z_tensor = torch.full(
                (num_agents,), z_t, dtype=torch.float32, device=self.device
            )

            c_all = policy.forward_policy(
                y_t, w_t, z_tensor, dist_tensor
            )
            c_all = torch.clamp(
                c_all, min=torch.zeros_like(w_t), max=w_t
            )
            c_detached = c_all.detach().clone()
            c_detached[sample_idx_t] = c_all[sample_idx_t]
            c_all = c_detached

            c_path.append(c_all[sample_idx_t].unsqueeze(-1))
            w_path.append(w_t[sample_idx_t].unsqueeze(-1))

            k_next = w_t - c_all

            eps_y = torch.from_numpy(
                rng.standard_normal(num_agents)
            ).float().to(self.device)
            eps_z = torch.tensor(
                float(rng.standard_normal()),
                device=self.device
            )

            y_next = rho_y * y_t + sigma_y * eps_y
            y_next = self._normalize_productivity_torch(y_next)
            z_next = rho_z * z_t + sigma_z * eps_z

            K_next = torch.sum(k_next)
            L_next = torch.sum(torch.exp(y_next))
            z_level = torch.exp(z_next)

            if float(K_next.item()) > 0 and float(L_next.item()) > 0:
                R_next = (
                    1 - delta +
                    z_level * alpha * (K_next ** (alpha - 1)) *
                    (L_next ** (1 - alpha))
                )
                W_next = (
                    z_level * (1 - alpha) *
                    (K_next ** alpha) * (L_next ** (-alpha))
                )
            else:
                R_next = 1 - delta
                W_next = z_level * (1 - alpha)

            w_next = R_next * k_next + W_next * torch.exp(y_next)

            w_t = w_next
            y_t = y_next
            z_t = z_next

        c_path_t = torch.stack(c_path, dim=0)
        w_path_t = torch.stack(w_path, dim=0)
        return self.objective_computer.lifetime_reward_objective(
            c_path_t, w_path_t
        )

    def _save_checkpoint(
        self,
        policy,
        objective_name: str,
        num_agents: int
    ):
        """Save policy checkpoint (Section 5.2)."""
        checkpoint_path = (
            self.checkpoint_dir /
            f"policy_{objective_name}_{num_agents}.pt"
        )
        torch.save(policy.state_dict(), checkpoint_path)

    def _save_objective_outputs(
        self,
        objective_name: str,
        results: List[Dict],
        simulations: Dict[int, Dict]
    ):
        """Save all outputs for a single objective."""
        plot_cfg = self.config.get('plotting', {})
        smoothing_window = int(plot_cfg.get('smoothing_window', 1))
        show_raw = bool(plot_cfg.get('show_raw', True))

        # 1. Generate plots for each agent count
        for result in results:
            num_agents = result['num_agents']
            metrics_df = result['metrics']
            simulation = result['simulation']
            policy = result['policy']

            # Create objective-specific plot
            objective_plot_dir = self.plots_dir / objective_name
            objective_plot_dir.mkdir(parents=True, exist_ok=True)
            self.plotter.plot_objective_results(
                metrics_df,
                simulation,
                objective_name,
                objective_plot_dir,
                policy=policy,
                model=self.model,
                num_agents=num_agents,
                smoothing_window=smoothing_window,
                show_raw=show_raw
            )

        # 2. Save metrics for this objective
        if results:
            combined_metrics = pd.concat(
                [r['metrics'] for r in results],
                ignore_index=True
            )
            metrics_path = (
                self.output_dir / f"metrics_{objective_name}.csv"
            )
            combined_metrics.to_csv(metrics_path, index=False)

    def _generate_cross_objective_outputs(self):
        """Generate cross-objective comparison (Section 5.4, 5.5)."""
        if len(self.all_simulations) < 2:
            print("Insufficient objectives for comparison")
            return

        # Get simulations for smallest agent count
        min_agents = min(
            min(sims.keys())
            for sims in self.all_simulations.values()
        )

        comparison_policies = {
            obj: policies[min_agents]
            for obj, policies in self.all_policies.items()
            if min_agents in policies
        }

        comparison_sims = {
            obj: sims[min_agents]
            for obj, sims in self.all_simulations.items()
            if min_agents in sims
        }

        if not comparison_policies:
            print("No policies available for comparison")
            return

        base_obj = (
            'lifetime_reward'
            if 'lifetime_reward' in comparison_sims
            else next(iter(comparison_sims))
        )
        base_sim = comparison_sims[base_obj]
        w_init = base_sim['w_path'][0]
        y_init = base_sim['y_path'][0]
        z_init = float(base_sim['z_path'][0])
        K_init = float(base_sim['K_path'][0])

        comparison_sims = {
            obj: self.evaluator.evaluate_simulation(
                policy,
                w_init,
                y_init,
                z_init,
                K_init,
                T=self.config['training'].get('eval_horizon', 1000),
                seed=self.config.get('seed', 42)
            )
            for obj, policy in comparison_policies.items()
        }

        if len(comparison_sims) >= 2:
            # Generate comparison figure
            self.plotter.plot_comparison_across_objectives(
                comparison_sims,
                self.comparison_dir,
                policies=comparison_policies,
                model=self.model,
                base_state={
                    'w_init': w_init,
                    'y_init': y_init,
                    'z_init': z_init
                }
            )

    def _save_comprehensive_tables(self):
        """Save comprehensive tables per section 5.4, 5.5."""
        if not self.all_results:
            return

        # Create comparison table (wide format)
        comparison_df = self.reporter.create_comparison_table(
            self.all_results,
            self.comparison_dir / 'table_all_objectives.csv'
        )

        # Also save as PNG
        self.reporter.save_comparison_table_as_png(
            comparison_df,
            self.comparison_dir / 'table_all_objectives.png'
        )

        # Create objective-specific tables
        self.reporter.create_objective_tables(
            self.all_results,
            self.tables_dir
        )

        # Print summary
        self.reporter.print_summary(self.all_results)

        print("\nTables saved to:")
        print(f"  {self.tables_dir}")
        print(f"  {self.comparison_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run Section 5 complete experiment'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/section5.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use'
    )

    args = parser.parse_args()

    runner = KSExperimentRunnerComplete(
        args.config, device=args.device
    )
    runner.run_full_experiment()


if __name__ == '__main__':
    main()

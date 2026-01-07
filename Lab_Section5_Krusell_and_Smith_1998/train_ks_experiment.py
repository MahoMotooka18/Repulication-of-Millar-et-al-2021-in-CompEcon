"""
Complete Training and Evaluation for Section 5 (Krusell-Smith)

Orchestrates training, evaluation, plotting, and table generation
according to section5_math.md requirements.
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
from typing import Dict, List, Tuple

from model_ks1998 import KrusellSmithModel, KrusellSmithParams
from nn_policy_ks import KSPolicyFactory
from objectives_ks import KSObjectiveComputer
from evaluator_ks import KSEvaluator
from plot_section5 import KSPlotter
from report_ks import KSReporter
from policy_utils_ks import (
    PolicyOutputType,
    NormalizationSpec,
    InputScaleSpec,
    resolve_policy_output_type,
    scale_inputs_numpy,
    scale_inputs_torch,
    build_dist_features_numpy,
    build_dist_features_torch,
    consumption_from_share_torch
)


class KSExperimentRunnerComplete:
    """Experiment runner with all outputs per section5_math.md."""

    def __init__(self, config_path: str, device: str = 'cpu'):
        """
        Initialize experiment runner.
        
        Args:
            config_path: path to YAML config
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.config = self._load_config(config_path)
        self.normalization_spec = self._build_normalization_spec()
        self.input_scale_spec, self.input_scale_snapshot = self._build_input_scale_spec()
        self.policy_output_types = self._resolve_policy_output_types()
        self.setup_directories()
        self._write_policy_definition_snapshot()
        self._write_input_scaler_snapshot()
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
        self.debug_dir = self.output_dir / 'debug'
        
        for d in [self.checkpoint_dir, self.plots_dir,
                  self.tables_dir, self.comparison_dir,
                  self.debug_dir]:
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
        self.evaluator = KSEvaluator(
            self.model,
            self.device,
            input_scale_spec=self.input_scale_spec,
            policy_output_type=PolicyOutputType.C_SHARE
        )

    def _build_normalization_spec(self) -> NormalizationSpec:
        """Build normalization spec from config."""
        norm_cfg = self.config.get('normalization', {})
        return NormalizationSpec(
            w_scale=float(norm_cfg.get('w_scale', 1.0)),
            w_shift=float(norm_cfg.get('w_shift', 0.0)),
            w_normalized=bool(norm_cfg.get('w_normalized', False)),
            c_scale=float(norm_cfg.get('c_scale', 1.0)),
            c_shift=float(norm_cfg.get('c_shift', 0.0)),
            c_normalized=bool(norm_cfg.get('c_normalized', False))
        )

    def _build_input_scale_spec(self) -> Tuple[InputScaleSpec, Dict]:
        """Build input scaling spec based on steady-state formulas."""
        cfg = self.config.get('input_scaling', {})
        enabled = bool(cfg.get('enabled', True))

        alpha = float(self.config['model']['alpha'])
        beta = float(self.config['model']['beta'])
        delta = float(self.config['model']['delta'])
        rho_y = float(self.config['model']['rho_y'])
        rho_z = float(self.config['model']['rho_z'])
        sigma_y = float(self.config['model']['sigma_y'])
        sigma_z = float(self.config['model']['sigma_z'])

        r_ss = 1.0 / beta
        denom = max(r_ss - 1.0 + delta, 1e-6)
        k_ss = (alpha / denom) ** (1.0 / (1.0 - alpha))
        w_ss = r_ss * k_ss + (1.0 - alpha) * (k_ss ** alpha)
        c_ss = (k_ss ** alpha) - delta * k_ss
        phi_ss = c_ss / w_ss if w_ss > 0 else 0.5

        y_scale = float(cfg.get('y_scale', 2.0 * sigma_y / np.sqrt(1.0 - rho_y**2)))
        z_scale = float(cfg.get('z_scale', 2.0 * sigma_z / np.sqrt(1.0 - rho_z**2)))
        w_max_mult = float(cfg.get('w_max_multiplier', 4.0))
        w_min = float(cfg.get('w_min', 0.0))
        w_max = float(cfg.get('w_max', w_max_mult * w_ss))

        spec = InputScaleSpec(
            y_scale=y_scale,
            z_scale=z_scale,
            w_min=w_min,
            w_max=w_max,
            w_steady=w_ss,
            enabled=enabled
        )
        snapshot = {
            'enabled': enabled,
            'applied_to': {
                'training': enabled,
                'evaluation': enabled,
                'plotting': enabled
            },
            'y_scale': y_scale,
            'z_scale': z_scale,
            'w_min': w_min,
            'w_max': w_max,
            'w_steady': w_ss,
            'phi_steady': phi_ss,
            'steady_state': {
                'R_ss': r_ss,
                'K_ss': k_ss,
                'W_ss': (1.0 - alpha) * (k_ss ** alpha),
                'w_ss': w_ss,
                'c_ss': c_ss
            }
        }
        return spec, snapshot

    def _write_input_scaler_snapshot(self):
        """Write input scaling snapshot for debugging."""
        snapshot_path = self.debug_dir / 'input_scaler_snapshot.json'
        with open(snapshot_path, 'w') as f:
            json.dump(self.input_scale_snapshot, f, indent=2)

    def _resolve_policy_output_types(self) -> Dict[str, str]:
        """Resolve per-objective policy output type."""
        cfg = self.config.get('policy_output_types', {})
        default_type = cfg.get('default', PolicyOutputType.C_SHARE)
        obj_cfg = {k: v for k, v in cfg.items() if k != 'default'}
        mapping = {}
        for obj_name in self.config['training']['objectives']:
            mapping[obj_name] = resolve_policy_output_type(
                obj_name, obj_cfg, default_type
            )
        return mapping

    def _write_policy_definition_snapshot(self):
        """Write policy definition and normalization snapshot."""
        snapshot = {
            'policy_output_types': self.policy_output_types,
            'normalization': self.normalization_spec.to_dict(),
            'unnormalize_formula': {
                'w_raw': 'w_norm * scale_w + shift_w',
                'c_raw_from_share': 'c_raw = share * w_raw',
                'c_raw_from_level': 'c_raw = c_norm * scale_c + shift_c'
            }
        }
        snapshot_path = self.debug_dir / 'policy_definition_snapshot.json'
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f, indent=2)

    def run_section5_experiment(self):
        """Run Section 5 experiment grid (Sections 5.3–5.6)."""
        objectives = self.config['training']['objectives']
        agent_counts = self.config['training']['agent_counts']

        print("\n" + "=" * 80)
        print("SECTION 5 EXPERIMENT GRID (SECTIONS 5.3–5.6)")
        print("=" * 80)
        print(f"Objectives: {objectives}")
        print(f"Agent counts: {agent_counts}")
        print(f"Grid size: {len(objectives)} × {len(agent_counts)}")
        print("=" * 80 + "\n")

        # Grid loop
        for obj_name in objectives:
            print(f"\n{'=' * 70}")
            print(f"OBJECTIVE: {obj_name.upper()}")
            print(f"{'=' * 70}")

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

        # Generate cross-objective comparison (Section 5.6)
        self._generate_cross_objective_outputs()

        # Final comprehensive tables (Section 5.4, 5.5)
        self._save_comprehensive_tables()

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)


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
        policy_output_type = self.policy_output_types.get(
            objective_name, PolicyOutputType.C_SHARE
        )
        if policy_output_type != PolicyOutputType.C_SHARE:
            raise ValueError(
                "Section 5 policy must output consumption share (c_share)."
            )

        # Initialize state
        rng = np.random.default_rng(self.config['seed'])
        w_init = np.full(num_agents, float(self.input_scale_snapshot['w_steady']))
        y_init = np.zeros(num_agents)
        z_init = 0.0
        k_ss = float(self.input_scale_snapshot['steady_state']['K_ss'])
        K_init = k_ss * float(num_agents)

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
        w_sampling_cfg = self.config['training'].get('w_training_sampling', {})
        w_sampling_enabled = bool(w_sampling_cfg.get('enabled', False))
        pretrain_value_iters = (
            self.config['training'].get('pretrain_value_iters', 0)
            if objective_name == 'bellman' else 0
        )
        total_updates = int(np.ceil(total_periods / max(1, train_every)))
        pretrain_updates = int(pretrain_value_iters)
        if pretrain_value_iters > total_updates and train_every > 1:
            pretrain_updates = int(np.ceil(pretrain_value_iters / train_every))
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

            y_scaled, w_scaled, z_scaled = scale_inputs_numpy(
                y_t, w_t, z_t, self.input_scale_spec
            )
            dist_vec = build_dist_features_numpy(y_scaled, w_scaled)

            y_tensor = torch.from_numpy(y_scaled).float().to(self.device)
            w_tensor = torch.from_numpy(w_scaled).float().to(self.device)
            w_raw_tensor = torch.from_numpy(w_t).float().to(self.device)
            z_tensor = torch.full(
                (num_agents,), z_scaled,
                dtype=torch.float32,
                device=self.device
            )
            dist_tensor = (
                torch.from_numpy(dist_vec)
                .float().to(self.device)
                .unsqueeze(0).expand(num_agents, -1)
            )

            with torch.no_grad():
                c_t = consumption_from_share_torch(
                    policy, y_tensor, w_tensor, z_tensor, dist_tensor, w_raw_tensor
                )
                c_t = torch.clamp(
                    c_t,
                    min=torch.zeros_like(w_raw_tensor),
                    max=w_raw_tensor
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

                # Training batch uses augmented w range if enabled
                y_train_use = y_t.copy()
                w_train_use = w_t.copy()
                if w_sampling_enabled:
                    w_min = float(self.input_scale_spec.w_min)
                    w_max = float(self.input_scale_spec.w_max)
                    w_train_use[sample_idx] = rng.uniform(
                        w_min, w_max, size=sample_size
                    )

                y_scaled_train, w_scaled_train, z_scaled_train = scale_inputs_numpy(
                    y_train_use, w_train_use, z_t, self.input_scale_spec
                )
                dist_train = build_dist_features_numpy(
                    y_scaled_train, w_scaled_train
                )

                y_train_tensor = torch.from_numpy(y_scaled_train).float().to(self.device)
                w_train_tensor = torch.from_numpy(w_scaled_train).float().to(self.device)
                w_train_raw_tensor = torch.from_numpy(w_train_use).float().to(self.device)
                z_scaled_value = float(z_scaled_train) if np.isscalar(z_scaled_train) else float(np.asarray(z_scaled_train).reshape(-1)[0])
                z_train_tensor = torch.full(
                    (num_agents,), z_scaled_value,
                    dtype=torch.float32,
                    device=self.device
                )
                dist_train_tensor = (
                    torch.from_numpy(dist_train)
                    .float().to(self.device)
                    .unsqueeze(0).expand(num_agents, -1)
                )

                y_sample = y_train_tensor[sample_idx]
                w_sample = w_train_tensor[sample_idx]
                w_raw_sample = w_train_raw_tensor[sample_idx]
                z_sample = z_train_tensor[sample_idx]
                dist_sample = dist_train_tensor[sample_idx]

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

                # Recompute next states for training batch using augmented w
                with torch.no_grad():
                    c_train_all = consumption_from_share_torch(
                        policy,
                        y_train_tensor,
                        w_train_tensor,
                        z_train_tensor,
                        dist_train_tensor,
                        w_train_raw_tensor
                    )
                c_train_np = c_train_all.cpu().numpy()
                k_next_full = w_train_use - c_train_np
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
                    w_train_use, c_train_np, y_next_full_1, R_next_1, W_next_1
                )
                w_next_full_2 = self.model.state_transition(
                    w_train_use, c_train_np, y_next_full_2, R_next_2, W_next_2
                )

                y_next_scaled_1, w_next_scaled_1, z_next_scaled_1 = scale_inputs_numpy(
                    y_next_full_1, w_next_full_1, z_next_1, self.input_scale_spec
                )
                y_next_scaled_2, w_next_scaled_2, z_next_scaled_2 = scale_inputs_numpy(
                    y_next_full_2, w_next_full_2, z_next_2, self.input_scale_spec
                )

                dist_next_1 = build_dist_features_numpy(
                    y_next_scaled_1, w_next_scaled_1
                )
                dist_next_2 = build_dist_features_numpy(
                    y_next_scaled_2, w_next_scaled_2
                )

                y_next_1_sample = torch.from_numpy(
                    y_next_scaled_1[sample_idx]
                ).float().to(self.device)
                y_next_2_sample = torch.from_numpy(
                    y_next_scaled_2[sample_idx]
                ).float().to(self.device)
                w_next_1_sample = torch.from_numpy(
                    w_next_scaled_1[sample_idx]
                ).float().to(self.device)
                w_next_2_sample = torch.from_numpy(
                    w_next_scaled_2[sample_idx]
                ).float().to(self.device)
                w_next_1_raw_sample = torch.from_numpy(
                    w_next_full_1[sample_idx]
                ).float().to(self.device)
                w_next_2_raw_sample = torch.from_numpy(
                    w_next_full_2[sample_idx]
                ).float().to(self.device)
                z_next_1_sample = torch.full(
                    (sample_size,), z_next_scaled_1,
                    dtype=torch.float32,
                    device=self.device
                )
                z_next_2_sample = torch.full(
                    (sample_size,), z_next_scaled_2,
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
                        rng,
                        w_sampling_enabled=w_sampling_enabled,
                        w_min=float(self.input_scale_spec.w_min),
                        w_max=float(self.input_scale_spec.w_max),
                        input_scale_spec=self.input_scale_spec
                    )
                elif objective_name == 'euler':
                    loss = (
                        self.objective_computer.euler_objective(
                            policy, y_sample, w_sample, z_sample,
                            dist_sample,
                            w_raw_sample,
                            y_next_1_sample, w_next_1_sample, z_next_1_sample,
                            dist_next_1_sample,
                            w_next_1_raw_sample,
                            y_next_2_sample, w_next_2_sample, z_next_2_sample,
                            dist_next_2_sample,
                            w_next_2_raw_sample,
                            R_next_1, R_next_2,
                            nu_h=self.config['training'].get('nu_h', 1.0),
                            input_scale_spec=self.input_scale_spec
                        )
                    )
                elif objective_name == 'bellman':
                    self._set_bellman_pretrain(
                        policy, update_step <= pretrain_updates
                    )
                    loss = (
                        self.objective_computer.bellman_objective(
                            policy, y_sample, w_sample, z_sample,
                            dist_sample,
                            w_raw_sample,
                            y_next_1_sample, w_next_1_sample, z_next_1_sample,
                            dist_next_1_sample,
                            w_next_1_raw_sample,
                            y_next_2_sample, w_next_2_sample, z_next_2_sample,
                            dist_next_2_sample,
                            w_next_2_raw_sample,
                            nu_h=self.config['training'].get('nu_h', 1.0),
                            nu=self.config['training'].get('nu', 1.0),
                            input_scale_spec=self.input_scale_spec
                        )
                    )

                loss.backward()
                optimizer.step()
                metric = {
                    'epoch': update_step,
                    'period': period + 1,
                    'update_step': update_step,
                    'objective_train': loss.item(),
                    'train_points': sample_size,
                    'test_euler_residual_mean': float('nan'),
                    'test_euler_residual_p50': float('nan'),
                    'test_euler_residual_p90': float('nan'),
                    'test_lifetime_reward_mean': float('nan'),
                    'aggregate_capital_mean': float('nan'),
                    'aggregate_capital_std': float('nan'),
                    'wall_time_sec': float('nan'),
                    'seed': self.config['seed'],
                    'net_size': self.config['training']['hidden_size'],
                    'objective_name': objective_name,
                    'num_agents': num_agents
                }

                if update_step % eval_interval == 0:
                    wall_time = time.time() - start_time
                    self.evaluator.policy_output_type = policy_output_type
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

                    metric.update({
                        'test_euler_residual_mean': euler_stats['euler_residual_mean'],
                        'test_euler_residual_p50': euler_stats['euler_residual_p50'],
                        'test_euler_residual_p90': euler_stats['euler_residual_p90'],
                        'test_lifetime_reward_mean': lr_stats['lifetime_reward_mean'],
                        'aggregate_capital_mean': stats['K_mean'],
                        'aggregate_capital_std': stats['K_std'],
                        'wall_time_sec': wall_time
                    })

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
        self.evaluator.policy_output_type = policy_output_type
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
        rng: np.random.Generator,
        w_sampling_enabled: bool,
        w_min: float,
        w_max: float,
        input_scale_spec: InputScaleSpec
    ) -> torch.Tensor:
        """Lifetime reward training step using a finite-horizon rollout."""
        num_agents = len(w_init)
        T = int(self.config['model'].get('horizon', self.model.params.horizon))

        y_t = torch.from_numpy(y_init).float().to(self.device)
        w_t = torch.from_numpy(w_init).float().to(self.device)
        z_t = torch.tensor(float(z_init), device=self.device)
        sample_idx_t = torch.from_numpy(sample_idx).long().to(self.device)

        if w_sampling_enabled:
            w_override = torch.from_numpy(
                rng.uniform(w_min, w_max, size=sample_idx.shape[0])
            ).float().to(self.device)
            w_t = w_t.clone()
            w_t[sample_idx_t] = w_override

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
            y_scaled, w_scaled, z_scaled = scale_inputs_torch(
                y_t, w_t, z_t, input_scale_spec
            )
            dist_vec = build_dist_features_torch(y_scaled, w_scaled)
            dist_tensor = dist_vec.unsqueeze(0).expand(num_agents, -1)
            z_scaled_value = float(z_scaled.item()) if isinstance(z_scaled, torch.Tensor) else float(z_scaled)
            z_tensor = torch.full(
                (num_agents,), z_scaled_value, dtype=torch.float32, device=self.device
            )

            c_all = consumption_from_share_torch(
                policy, y_scaled, w_scaled, z_tensor, dist_tensor, w_t
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
            # Treat prices as exogenous to any single agent's choice.
            K_price = K_next.detach()
            L_price = L_next.detach()
            z_level = torch.exp(z_next.detach())

            if float(K_price.item()) > 0 and float(L_price.item()) > 0:
                R_next = (
                    1 - delta +
                    z_level * alpha * (K_price ** (alpha - 1)) *
                    (L_price ** (1 - alpha))
                )
                W_next = (
                    z_level * (1 - alpha) *
                    (K_price ** alpha) * (L_price ** (-alpha))
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
        w_plot_max = plot_cfg.get('w_plot_max')
        mismatch_cfg = self.config.get('mismatch_checks', {})
        max_agents = max(r['num_agents'] for r in results)
        policy_output_type = self.policy_output_types.get(
            objective_name, PolicyOutputType.C_SHARE
        )

        # 1. Generate plots for each agent count
        for result in results:
            num_agents = result['num_agents']
            metrics_df = result['metrics']
            simulation = result['simulation']
            policy = result['policy']

            # Create objective-specific plot
            objective_plot_dir = self.plots_dir / objective_name
            objective_plot_dir.mkdir(parents=True, exist_ok=True)
            plot_result = self.plotter.plot_objective_results(
                metrics_df,
                simulation,
                objective_name,
                objective_plot_dir,
                policy=policy,
                model=self.model,
                num_agents=num_agents,
                smoothing_window=smoothing_window,
                show_raw=show_raw,
                policy_output_type=policy_output_type,
                normalization_spec=self.normalization_spec,
                input_scale_spec=self.input_scale_spec,
                mismatch_config=mismatch_cfg,
                w_plot_max=w_plot_max,
                loss_scale_ref_points=plot_cfg.get(
                    'lifetime_reward_loss_ref_points', 10.0
                )
            )

            if num_agents == max_agents:
                self._write_plot_inputs_snapshot(
                    objective_name,
                    plot_result.get('policy_plot', {})
                )
                self._append_mismatch_checks(
                    plot_result.get('mismatch_checks', [])
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

        self.evaluator.policy_output_type = PolicyOutputType.C_SHARE
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
                },
                policy_output_types=self.policy_output_types,
                normalization_spec=self.normalization_spec,
                input_scale_spec=self.input_scale_spec,
                w_plot_max=self.config.get('plotting', {}).get('w_plot_max')
            )

    def _write_plot_inputs_snapshot(self, objective_name: str, policy_plot: Dict):
        """Save plot inputs snapshot for debugging."""
        if not policy_plot:
            return
        snapshot_path = self.debug_dir / 'plot_inputs_snapshot.npz'
        payload = {
            f"{objective_name}_w_grid_raw": policy_plot['w_grid_raw'],
            f"{objective_name}_y_grid_raw": policy_plot['y_grid_raw'],
            f"{objective_name}_y_grid_std": policy_plot['y_grid_std'],
            f"{objective_name}_c_grid_raw": policy_plot['c_grid_raw'],
            f"{objective_name}_c_share_grid": policy_plot['c_share_grid'],
            f"{objective_name}_steady_state": np.array([
                policy_plot['steady_state']['steady_y'],
                policy_plot['steady_state']['steady_w'],
                policy_plot['steady_state']['steady_z']
            ])
        }
        if snapshot_path.exists():
            existing = dict(np.load(snapshot_path, allow_pickle=True))
            existing.update(payload)
            np.savez(snapshot_path, **existing)
        else:
            np.savez(snapshot_path, **payload)

    def _append_mismatch_checks(self, checks: List[Dict]):
        """Append mismatch checks to jsonl file."""
        if not checks:
            return
        mismatch_path = self.debug_dir / 'mismatch_checks.jsonl'
        with open(mismatch_path, 'a') as f:
            for check in checks:
                f.write(json.dumps(check) + "\n")

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
        description='Run Section 5 experiments'
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
    runner.run_section5_experiment()


if __name__ == '__main__':
    main()

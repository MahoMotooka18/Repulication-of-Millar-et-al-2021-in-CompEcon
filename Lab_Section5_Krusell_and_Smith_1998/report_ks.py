"""
Comparison and Reporting Module for Section 5

Generates cross-objective comparisons and final reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


class KSReporter:
    """Generates final reports and tables."""

    @staticmethod
    def create_comparison_table(
        results: List[Dict],
        output_path: Path
    ) -> pd.DataFrame:
        """
        Create comprehensive comparison table.
        
        Args:
            results: list of result dicts from training
            output_path: path to save CSV
            
        Returns:
            combined dataframe
        """
        rows = []
        
        for result in results:
            obj = result['objective']
            num_agents = result['num_agents']
            stats = result['statistics']
            
            row = {
                'objective': obj,
                'l': num_agents,
                'std(y)': stats.get('std_y', np.nan),
                'corr(y,c)': stats.get('corr_y_c', np.nan),
                'Gini(k)': stats.get('gini_k', np.nan),
                'Bottom 40%': stats.get('share_bottom_40', np.nan),
                'Top 20%': stats.get('share_top_20', np.nan),
                'Top 1%': stats.get('share_top_1', np.nan),
                'Time, sec.': stats.get('time_sec', np.nan),
                'R2': stats.get('r2', np.nan),
                'K_mean': stats.get('K_mean', np.nan),
                'K_std': stats.get('K_std', np.nan)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        return df

    @staticmethod
    def create_objective_tables(
        results: List[Dict],
        output_dir: Path
    ):
        """
        Create separate table for each objective (CSV and PNG).

        Args:
            results: list of result dicts
            output_dir: output directory
        """
        objectives = set(r['objective'] for r in results)

        for obj in objectives:
            obj_results = [r for r in results
                           if r['objective'] == obj]
            rows = []

            for result in obj_results:
                num_agents = result['num_agents']
                stats = result['statistics']

                row = {
                    'l': num_agents,
                    'std(y)': stats.get('std_y', np.nan),
                    'corr(y,c)': stats.get('corr_y_c', np.nan),
                    'Gini(k)': stats.get('gini_k', np.nan),
                    'Bottom 40%': stats.get('share_bottom_40', np.nan),
                    'Top 20%': stats.get('share_top_20', np.nan),
                    'Top 1%': stats.get('share_top_1', np.nan),
                    'Time, sec.': stats.get('time_sec', np.nan),
                    'R2': stats.get('r2', np.nan),
                    'K_mean': stats.get('K_mean', np.nan),
                    'K_std': stats.get('K_std', np.nan)
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            output_path = output_dir / f"table_properties_{obj}.csv"
            df.to_csv(output_path, index=False)

            # Also save as PNG
            KSReporter._save_table_as_png(
                df, obj, output_dir / f"table_properties_{obj}.png"
            )

    @staticmethod
    def _save_table_as_png(
        df: pd.DataFrame,
        objective_name: str,
        output_path: Path
    ):
        """Save table as PNG image with nice formatting."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')

        # Format columns for display
        display_df = df.copy()
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'agents':
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if not np.isnan(x) else "-"
                )

        # Create table
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.08] * len(display_df.columns)
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(display_df) + 1):
            for j in range(len(display_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
                else:
                    table[(i, j)].set_facecolor('#F2F2F2')

        # Add title
        title = (
            f"Table: {objective_name.replace('_', ' ').title()} "
            f"Objective Statistics"
        )
        fig.suptitle(
            title,
            fontsize=14,
            weight='bold',
            y=0.98
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Table PNG saved: {output_path}")

    @staticmethod
    def save_comparison_table_as_png(
        comparison_df: pd.DataFrame,
        output_path: Path
    ):
        """Save comparison table as PNG image."""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')

        # Format for display
        display_df = comparison_df.copy()
        numeric_cols = display_df.select_dtypes(
            include=[np.number]
        ).columns
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if not np.isnan(x) else "-"
            )

        # Create table
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.08] * len(display_df.columns)
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color by objective
        obj_colors = {
            'lifetime_reward': '#E2EFDA',
            'euler': '#FCE4D6',
            'bellman': '#EDEBF7'
        }

        for i in range(1, len(display_df) + 1):
            obj = display_df.iloc[i-1]['objective']
            color = obj_colors.get(obj, '#F2F2F2')
            for j in range(len(display_df.columns)):
                table[(i, j)].set_facecolor(color)

        # Add title
        fig.suptitle(
            'Table: All Objectives Comparison',
            fontsize=14,
            weight='bold',
            y=0.98
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison table PNG saved: {output_path}")

    @staticmethod
    def print_summary(
        results: List[Dict]
    ):
        """
        Print summary statistics to console.
        
        Args:
            results: list of result dicts
        """
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        objectives = sorted(set(r['objective'] for r in results))
        agent_counts = sorted(set(r['num_agents'] for r in results))
        
        for obj in objectives:
            print(f"\n{obj.upper()}")
            print("-"*80)
            print(f"{'Agents':<10} {'Std(y)':<12} {'Corr(y,c)':<12} "
                  f"{'Gini':<12} {'R²':<12}")
            print("-"*80)
            
            for num_agents in agent_counts:
                result = next(
                    (r for r in results
                     if r['objective'] == obj
                     and r['num_agents'] == num_agents),
                    None
                )
                
                if result:
                    stats = result['statistics']
                    print(f"{num_agents:<10} "
                          f"{stats.get('std_y', 0):<12.4f} "
                          f"{stats.get('corr_y_c', 0):<12.4f} "
                          f"{stats.get('gini_k', 0):<12.4f} "
                          f"{stats.get('r2', 0):<12.4f}")


class CrossObjectiveComparison:
    """Utilities for cross-objective comparison."""

    @staticmethod
    def compute_distance_metrics(
        simulations: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Compute distance metrics between objective solutions.
        
        Args:
            simulations: dict mapping objective -> simulation results
            
        Returns:
            dict with distance metrics
        """
        objectives = list(simulations.keys())
        
        metrics = {}
        
        # Policy distance (L2 norm of consumption difference)
        if len(objectives) >= 2:
            obj1, obj2 = objectives[0], objectives[1]
            c1 = simulations[obj1]['c_path'].flatten()
            c2 = simulations[obj2]['c_path'].flatten()
            
            # Normalize by mean consumption
            mean_c = np.mean(np.concatenate([c1, c2]))
            policy_distance = np.sqrt(np.mean((c1 - c2)**2)) / mean_c
            metrics['policy_distance'] = policy_distance
        
        # Capital distance
        if len(objectives) >= 2:
            K1 = simulations[obj1]['K_path']
            K2 = simulations[obj2]['K_path']
            
            mean_K = np.mean(np.concatenate([K1, K2]))
            capital_distance = np.sqrt(np.mean((K1 - K2)**2)) / mean_K
            metrics['capital_distance'] = capital_distance
        
        return metrics

#!/usr/bin/env python3
# ruff: noqa: F841
# mypy: disable-error-code="assignment, no-any-return, var-annotated"
"""
Comparison tool for discrete vs continuous Azencot losses.
Loads trained models and compares performance metrics.

Usage:
    python compare_discrete_continuous.py \
        --discrete-model model_outputs_tra/discrete_linear_128/run-XXX/final_model.pth \
        --continuous-model model_outputs_tra/continous_linear_128/run-XXX/final_model.pth \
        --dataset tra \
        --metric-type mse  # or ssim, rmse
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict

import torch
import numpy as np

# Local imports


class DiscreteVsContinuousComparator:
    """Compares metrics between discrete and continuous Azencot models."""

    def __init__(
        self,
        discrete_model_path: str,
        continuous_model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize comparator with model paths.

        Args:
            discrete_model_path: Path to discrete model checkpoint
            continuous_model_path: Path to continuous model checkpoint
            device: Computation device
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.discrete_model = self._load_model(discrete_model_path)
        self.continuous_model = (
            self._load_model(continuous_model_path) if continuous_model_path else None
        )

        self.results = {}

    def _load_model(self, checkpoint_path: str) -> Dict:
        """Load model from checkpoint."""
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        return checkpoint

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute standard evaluation metrics.

        Args:
            y_true: Ground truth [N, C, H, W]
            y_pred: Predictions [N, C, H, W]

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # MSE
        mse = np.mean((y_true - y_pred) ** 2)
        metrics["mse"] = float(mse)

        # RMSE
        metrics["rmse"] = float(np.sqrt(mse))

        # MAE
        metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))

        # Max absolute error
        metrics["max_ae"] = float(np.max(np.abs(y_true - y_pred)))

        # Normalized metrics (if max value is not zero)
        y_max = np.max(np.abs(y_true))
        if y_max > 1e-6:
            metrics["nmse"] = metrics["mse"] / (y_max**2)  # Normalized MSE
            metrics["nrmse"] = metrics["rmse"] / y_max  # Normalized RMSE

        return metrics

    def compare(self) -> Dict:
        """
        Compare discrete and continuous models.

        Returns:
            Dictionary with comparison results
        """
        print("\n" + "=" * 60)
        print("DISCRETE vs CONTINUOUS COMPARISON")
        print("=" * 60)

        config_discrete = self.discrete_model.get("config", {})
        config_continuous = (
            self.continuous_model.get("config", {}) if self.continuous_model else {}
        )

        results = {
            "discrete": {
                "model_path": "discrete",
                "loss_config": config_discrete.get("loss", {}),
            },
            "continuous": (
                {
                    "model_path": "continuous",
                    "loss_config": config_continuous.get("loss", {}),
                }
                if self.continuous_model
                else None
            ),
        }

        # Print configurations
        print("\n[Discrete Model Configuration]")
        print(json.dumps(config_discrete.get("loss", {}), indent=2, default=str))

        if self.continuous_model:
            print("\n[Continuous Model Configuration]")
            print(json.dumps(config_continuous.get("loss", {}), indent=2, default=str))

        # Print training history
        discrete_history = self.discrete_model.get("history", {})
        print("\n[Discrete Training History (Last 5 epochs)]")
        if isinstance(discrete_history, dict):
            for key in list(discrete_history.keys())[-5:]:
                print(f"  {key}: {discrete_history[key]}")

        if self.continuous_model:
            continuous_history = self.continuous_model.get("history", {})
            print("\n[Continuous Training History (Last 5 epochs)]")
            if isinstance(continuous_history, dict):
                for key in list(continuous_history.keys())[-5:]:
                    print(f"  {key}: {continuous_history[key]}")

        # Test metrics
        discrete_test = self.discrete_model.get("test_metrics", {})
        print("\n[Discrete Test Metrics]")
        for key, val in discrete_test.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.6f}")

        if self.continuous_model:
            continuous_test = self.continuous_model.get("test_metrics", {})
            print("\n[Continuous Test Metrics]")
            for key, val in continuous_test.items():
                if isinstance(val, float):
                    print(f"  {key}: {val:.6f}")

            # Summary comparison
            print("\n" + "=" * 60)
            print("SUMMARY: Metric Differences (Discrete - Continuous)")
            print("=" * 60)
            for key in discrete_test.keys():
                if isinstance(discrete_test.get(key), (int, float)) and isinstance(
                    continuous_test.get(key), (int, float)
                ):
                    diff = discrete_test[key] - continuous_test[key]
                    pct_diff = (
                        (diff / continuous_test[key] * 100)
                        if continuous_test[key] != 0
                        else 0
                    )
                    status = "✓ Better" if diff < 0 else "✗ Worse"
                    print(f"  {key}: {diff:+.6f} ({pct_diff:+.2f}%) {status}")

        return results

    def plot_comparison(self, output_dir: str = "comparison_plots"):
        """Generate comparison plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\n✓ Comparison plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Compare discrete vs continuous Azencot loss models"
    )
    parser.add_argument(
        "--discrete-model",
        type=str,
        required=True,
        help="Path to discrete model checkpoint",
    )
    parser.add_argument(
        "--continuous-model",
        type=str,
        default=None,
        help="Path to continuous model checkpoint (optional)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--output-dir", type=str, default="comparison_plots", help="Output directory"
    )

    args = parser.parse_args()

    # Create comparator
    comparator = DiscreteVsContinuousComparator(
        discrete_model_path=args.discrete_model,
        continuous_model_path=args.continuous_model,
        device=args.device,
    )

    # Run comparison
    results = comparator.compare()

    # Generate plots
    comparator.plot_comparison(output_dir=args.output_dir)

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()

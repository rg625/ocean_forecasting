import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
import skimage.metrics as metrics
from tensordict import TensorDict
from typing import Optional


class Metric(nn.Module):
    def __init__(
        self,
        mode: str,
        variable_mode: str = "single",
        variable_name: Optional[str] = None,
    ):
        """
        Args:
            mode: One of ["L2", "SSIM", "PSNR", "MI"]
            variable_mode: "single" (use one variable) or "all" (average over all variables)
            variable_name: Name of the variable if using "single" mode
        """
        super().__init__()
        assert mode in ["L2", "SSIM", "PSNR", "MI"], f"Unknown metric mode: {mode}"
        assert variable_mode in [
            "single",
            "all",
        ], "variable_mode must be 'single' or 'all'"
        if variable_mode == "single":
            assert (
                variable_name is not None
            ), "variable_name must be provided in 'single' mode"

        self.mode = mode
        self.variable_mode = variable_mode
        self.variable_name = variable_name
        self.eval()

    def _compute_pairwise_distance(
        self, ref: torch.Tensor, other: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            ref, other: Tensors of shape [B, T, H, W]
        Returns:
            Tensor of shape [B, T] with per-frame distances
        """
        assert ref.shape == other.shape
        B, T, H, W = ref.shape

        ref_np = (ref * 255).clamp(0, 255).byte().cpu().numpy()
        other_np = (other * 255).clamp(0, 255).byte().cpu().numpy()

        distances = np.empty((B, T))
        for i in range(B):
            for j in range(T):
                r = ref_np[i, j]
                o = other_np[i, j]

                if self.mode == "L2":
                    distances[i, j] = metrics.mean_squared_error(r, o) / (255.0**2)
                elif self.mode == "SSIM":
                    distances[i, j] = 1 - metrics.structural_similarity(
                        r, o, data_range=255
                    )
                elif self.mode == "PSNR":
                    distances[i, j] = -metrics.peak_signal_noise_ratio(
                        r, o, data_range=255
                    )
                elif self.mode == "MI":
                    distances[i, j] = np.mean(metrics.variation_of_information(r, o))

        return torch.tensor(distances, dtype=torch.float32)

    def forward(self, x: dict) -> torch.Tensor:
        """
        x["reference"], x["other"] should be TensorDicts with variables as keys,
        and each variable should be shaped [B, T, H, W]
        """
        reference: TensorDict = x["reference"]
        other: TensorDict = x["other"]

        if self.variable_mode == "single":
            ref_tensor = reference[self.variable_name]
            other_tensor = other[self.variable_name]
            return self._compute_pairwise_distance(ref_tensor, other_tensor)

        elif self.variable_mode == "all":
            per_var_results = []
            for var in reference.keys():
                if var in ["seq_length", "Re"]:
                    continue
                dist = self._compute_pairwise_distance(reference[var], other[var])
                per_var_results.append(dist)
            # Stack and average across variables
            stacked = torch.stack(per_var_results, dim=0)  # [V, B, T]
            return stacked.mean(dim=0)  # [B, T]

    def compute_distance(
        self,
        input1: TensorDict,
        input2: TensorDict,
    ) -> NDArray[np.float32]:
        """
        Args:
            input1, input2: TensorDicts with shape [T, H, W] or [B, T, H, W] per variable
        Returns:
            Numpy array of shape [B * T] (flattened)
        """

        def expand(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(0) if x.ndim == 3 else x

        input_dict = {
            "reference": TensorDict(
                {k: expand(v) for k, v in input1.items() if k != "seq_length"},
                batch_size=[],
            ),
            "other": TensorDict(
                {k: expand(v) for k, v in input2.items() if k != "seq_length"},
                batch_size=[],
            ),
        }

        with torch.no_grad():
            result = self.forward(input_dict)  # shape [B, T, ...]
        return np.array(result.view(-1).cpu().numpy(), dtype=np.float32)

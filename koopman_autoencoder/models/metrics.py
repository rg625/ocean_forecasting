# models/metrics.py
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
import skimage.metrics as sk_metrics
from tensordict import TensorDict
from typing import Optional, List


class Metric(nn.Module):
    """
    Computes image-based comparison metrics (L2, SSIM, PSNR, VI).
    Designed to work with TensorDicts in a batch-wise fashion.
    """

    VALID_MODES = ["L2", "SSIM", "PSNR", "VI"]

    def __init__(
        self,
        mode: str,
        variable_mode: str = "single",
        variable_name: Optional[str] = None,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown metric mode '{mode}'. Valid modes: {self.VALID_MODES}"
            )
        if variable_mode not in ["single", "all"]:
            raise ValueError("variable_mode must be 'single' or 'all'")
        if variable_mode == "single" and not variable_name:
            raise ValueError("variable_name must be provided for 'single' mode")

        self.mode = mode
        self.variable_mode = variable_mode
        self.variable_name = variable_name
        self.eval()  # Set to evaluation mode by default

    def _compute_distance(self, ref: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Computes pairwise distance for a single variable's tensors."""
        assert ref.shape == other.shape and ref.ndim == 4, "Inputs must be [B, T, H, W]"
        B, T, H, W = ref.shape

        # Prepare for skimage: move to CPU, scale to [0, 255] uint8
        ref_np = (ref * 255).clamp(0, 255).byte().cpu().numpy()
        other_np = (other * 255).clamp(0, 255).byte().cpu().numpy()

        distances = np.zeros((B, T), dtype=np.float32)
        for i in range(B):
            for j in range(T):
                r, o = ref_np[i, j], other_np[i, j]
                if self.mode == "L2":
                    distances[i, j] = sk_metrics.mean_squared_error(r, o) / (255.0**2)
                elif self.mode == "SSIM":
                    # SSIM is a similarity, 1 - SSIM is a distance
                    distances[i, j] = 1 - sk_metrics.structural_similarity(
                        r, o, data_range=255
                    )
                elif self.mode == "PSNR":
                    # PSNR is inverted to act as a distance (lower is better)
                    distances[i, j] = -sk_metrics.peak_signal_noise_ratio(
                        r, o, data_range=255
                    )
                elif self.mode == "VI":
                    # Variation of Information
                    distances[i, j] = np.mean(sk_metrics.variation_of_information(r, o))

        return torch.from_numpy(distances)

    def forward(self, reference: TensorDict, other: TensorDict) -> torch.Tensor:
        """
        Computes metric between two TensorDicts.

        Args:
            reference (TensorDict): The ground truth data.
            other (TensorDict): The predicted data.

        Returns:
            torch.Tensor: A tensor of shape [B, T] with the computed distances.
        """
        if self.variable_mode == "single":
            ref_tensor = reference.get(self.variable_name).unsqueeze(
                2
            )  # Add channel dim
            other_tensor = other.get(self.variable_name).unsqueeze(2)
            return self._compute_distance(ref_tensor, other_tensor)

        elif self.variable_mode == "all":
            per_var_results: List[torch.Tensor] = []
            valid_keys = [
                k
                for k in reference.keys()
                if k not in ["seq_length", "Re", "obstacle_mask"]
            ]
            for var in valid_keys:
                ref_tensor = reference.get(var)
                other_tensor = other.get(var)
                # Ensure tensors have a channel dimension for consistency
                if ref_tensor.ndim == 3:
                    ref_tensor = ref_tensor.unsqueeze(2)
                if other_tensor.ndim == 3:
                    other_tensor = other_tensor.unsqueeze(2)

                dist = self._compute_distance(ref_tensor, other_tensor)
                per_var_results.append(dist)

            if not per_var_results:
                return torch.empty(0)
            # Stack and average across variables
            return torch.stack(per_var_results, dim=0).mean(dim=0)

    def compute_distance(
        self, input1: TensorDict, input2: TensorDict
    ) -> NDArray[np.float32]:
        """Convenience wrapper for use outside of the training loop."""

        def _prepare_td(td: TensorDict) -> TensorDict:
            # Ensure all tensors are 4D [B, T, H, W] for processing
            out = {}
            # Infer batch size from a tensor
            b_size = next(iter(td.values())).shape[0] if td.batch_size else 1
            for k, v in td.items():
                if k in ["seq_length", "Re", "obstacle_mask"]:
                    continue
                if v.ndim == 2:  # [H, W] -> [1, 1, H, W]
                    out[k] = v.unsqueeze(0).unsqueeze(0)
                elif (
                    v.ndim == 3
                ):  # [T, H, W] or [B, H, W]? Assume [T, H, W] -> [1, T, H, W]
                    out[k] = v.unsqueeze(0)
            return TensorDict(out, batch_size=[b_size])

        with torch.no_grad():
            result = self.forward(_prepare_td(input1), _prepare_td(input2))
        return np.array(result.view(-1).cpu().numpy())

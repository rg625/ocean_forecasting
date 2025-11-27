import torch
import torch.nn as nn
import numpy as np
import skimage.metrics as sk_metrics
from tensordict import TensorDict
from typing import Optional, List
from pathlib import Path
import logging

# LSIM is an optional dependency
try:
    from .lsim.distance_model import DistanceModel

    LSIM_AVAILABLE = True
except ImportError:
    DistanceModel = None  # type: ignore[assignment, misc]
    LSIM_AVAILABLE = False
    logging.warning("LSIM models not found. LSIM metric will not be available.")


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class Metric(nn.Module):
    """
    Computes image-based comparison metrics between two sets of video data
    represented as TensorDicts.

    This module supports standard image metrics (L1, L2, SSIM, PSNR, VI) and
    learned perceptual metrics (LSIM).

    The `forward` method returns a tensor of distances.
    - If `variable_mode='single'`, returns shape [B, T].
    - If `variable_mode='all'`, returns shape [B, T, C], where C is the
      number of variables (channels) being compared.

    This output shape is designed to match the dimensional averaging logic
    of the original evaluation snippets (e.g., `mseScalar`, `lsimScalar`).
    """

    VALID_MODES = ["L1", "L2", "SSIM", "PSNR", "VI", "LSIM"]
    SKIMAGE_MODES = ["SSIM", "PSNR", "VI"]  # L1/L2 are handled in PyTorch
    LSIM_MODES = ["LSIM"]
    PYTORCH_MODES = ["L1", "L2"]

    def __init__(
        self,
        mode: str,
        variable_mode: str = "single",
        variable_name: Optional[str] = None,
        lsim_model_path: Optional[str | Path] = "./models/lsim/models/LSiM.pth",
    ):
        super().__init__()
        # --- Validate Inputs ---
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown metric mode '{mode}'. Valid modes: {self.VALID_MODES}"
            )
        if variable_mode not in ["single", "all"]:
            raise ValueError("`variable_mode` must be 'single' or 'all'")
        if variable_mode == "single" and not variable_name:
            raise ValueError("`variable_name` must be provided for 'single' mode")

        if mode in self.LSIM_MODES and not LSIM_AVAILABLE:
            raise ImportError(
                "LSIM metric selected, but LSIM dependencies are not installed or found."
            )
        if mode in self.LSIM_MODES and not lsim_model_path:
            raise ValueError("`lsim_model_path` must be provided for LSIM modes")

        self.mode = mode
        self.variable_mode = variable_mode
        self.variable_name = variable_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Setup LSIM Model if required ---
        if self.mode in self.LSIM_MODES:
            assert (
                lsim_model_path is not None
            ), "LSIM model path must be provided for LSIM modes"
            self._setup_lsim(Path(lsim_model_path))

        self.eval()  # Set to evaluation mode by default

    def _setup_lsim(self, model_path: Path):
        """
        Loads the LSIM model using the high-level DistanceModel wrapper.
        """
        logger.info(f"Setting up {self.mode} model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"LSIM model weights not found at: {model_path}")

        use_gpu = self.device.type == "cuda"
        self.lsim_model = DistanceModel(baseType="lsim", isTrain=False, useGPU=use_gpu)
        self.lsim_model.load(str(model_path))

        # Set to eval mode and freeze weights, as in the reference loss script
        self.lsim_model.eval()
        for param in self.lsim_model.parameters():
            param.requires_grad = False

        logger.info("LSIM DistanceModel loaded and frozen.")

    def _compute_lsim_distance(
        self, ref: torch.Tensor, other: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the LSIM distance using the DistanceModel wrapper,
        replicating the vectorized logic from the reference `loss_lsim`.

        Args:
            ref (torch.Tensor): Ground truth tensor of shape [B, T, H, W].
            other (torch.Tensor): Predicted tensor of shape [B, T, H, W].

        Returns:
            torch.Tensor: A tensor of shape [B, T] with the LSIM distance
                          for each frame.
        """
        assert (
            ref.ndim == 4
        ), f"Input tensors must be 4D [B, T, H, W], but got shape {ref.shape}"
        ref = ref.to(self.device)
        other = other.to(self.device)
        B, T, H, W = ref.shape

        # Add a channel dimension: [B, T, 1, H, W]
        ref_5d = ref.unsqueeze(2)
        other_5d = other.unsqueeze(2)

        # --- 1. Per-sequence normalization to [0, 255] ---
        # Normalization is per-sequence (B) across T, H, W.
        ref_min = torch.amin(ref_5d, dim=(1, 3, 4), keepdim=True)
        ref_max = torch.amax(ref_5d, dim=(1, 3, 4), keepdim=True)
        other_min = torch.amin(other_5d, dim=(1, 3, 4), keepdim=True)
        other_max = torch.amax(other_5d, dim=(1, 3, 4), keepdim=True)

        # Use a combined min/max for both tensors to ensure same scaling
        all_min = torch.min(ref_min, other_min)
        all_max = torch.max(ref_max, other_max)

        epsilon = 1e-6  # Prevent division by zero
        ref_norm = 255 * (ref_5d - all_min) / (all_max - all_min + epsilon)
        other_norm = 255 * (other_5d - all_min) / (all_max - all_min + epsilon)

        # --- 2. Reshape and Expand for LSIM model ---
        # Flatten B, T into the batch dimension.
        # (B, T, 1, H, W) -> (B*T, 1, H, W)
        ref_reshaped = ref_norm.reshape(-1, 1, H, W)
        other_reshaped = other_norm.reshape(-1, 1, H, W)

        # Expand the channel dimension to 3, as required by the LSIM model.
        # -> [B*T, 1, 3, H, W]
        ref_expanded = ref_reshaped.expand(-1, 3, -1, -1).unsqueeze(
            1
        )  # [B*T, 1, 3, H, W]
        other_expanded = other_reshaped.expand(-1, 3, -1, -1).unsqueeze(
            1
        )  # [B*T, 1, 3, H, W]

        # --- 3. Compute LSIM distance ---
        # Pass tensors directly to the model (as it's an nn.Module)
        in_dict = {"reference": ref_expanded, "other": other_expanded}

        with torch.no_grad():
            # Model returns a flat tensor of distances, shape [B*T]
            distances_flat = self.lsim_model(in_dict)

        # --- 4. Reshape output ---
        # Reshape flat distances [B*T] back to [B, T]
        distances_bt = distances_flat.reshape(B, T)

        return distances_bt.to(ref.device)

    def _compute_skimage_distance(
        self, ref: torch.Tensor, other: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes scikit-image metrics (SSIM, PSNR, VI) or fast PyTorch
        metrics (L1, L2) between two 4D tensors [B, T, H, W].

        This function computes metrics on the *raw data* without normalization,
        matching the logic of F.mse_loss.
        """
        assert ref.ndim == 4, f"Input tensors must be 4D, but got shape {ref.shape}"

        # --- PYTORCH MODES (L1, L2): Fast, GPU-native ---
        # This matches `F.mse_loss` and `F.l1_loss` logic
        if self.mode == "L2":
            # Compute MSE (L2) per-frame by averaging over spatial dimensions (H, W)
            # Input: [B, T, H, W] -> Output: [B, T]
            return torch.mean((ref - other) ** 2, dim=(-2, -1))

        if self.mode == "L1":
            # Compute MAE (L1) per-frame
            # Input: [B, T, H, W] -> Output: [B, T]
            return torch.mean(torch.abs(ref - other), dim=(-2, -1))

        # --- SKIMAGE MODES (SSIM, PSNR, VI): Slow, requires CPU ---
        B, T, H, W = ref.shape

        # Move to CPU and convert to numpy once for efficiency
        ref_np = ref.cpu().numpy()
        other_np = other.cpu().numpy()

        distances = np.zeros((B, T), dtype=np.float32)

        for i in range(B):
            for j in range(T):
                r, o = ref_np[i, j], other_np[i, j]

                # --- Robust data_range calculation for non-normalized data ---
                r_min, r_max = r.min(), r.max()
                o_min, o_max = o.min(), o.max()
                data_range = max(r_max, o_max) - min(r_min, o_min)
                if data_range < 1e-6:
                    # If data range is zero (e.g., all black), distance is zero
                    distances[i, j] = 0.0
                    continue

                if self.mode == "SSIM":
                    # SSIM is a similarity [0, 1], so 1 - SSIM is a distance
                    distances[i, j] = 1.0 - sk_metrics.structural_similarity(
                        r, o, data_range=data_range
                    )
                elif self.mode == "PSNR":
                    # PSNR is inverted to act as a distance (lower is better)
                    mse = sk_metrics.mean_squared_error(r, o)
                    if mse < 1e-9:
                        psnr = 100.0  # Arbitrarily high for identical images
                    else:
                        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
                    distances[i, j] = -psnr
                elif self.mode == "VI":
                    # VI requires integer inputs. We scale to [0, 255] based on
                    # the dynamic data range for robust comparison.
                    r_norm = (r - min(r_min, o_min)) / data_range
                    o_norm = (o - min(r_min, o_min)) / data_range

                    r_int = (r_norm * 255).astype(np.uint8)
                    o_int = (o_norm * 255).astype(np.uint8)

                    distances[i, j] = np.mean(
                        sk_metrics.variation_of_information(r_int, o_int)
                    )

        return torch.from_numpy(distances).to(ref.device)

    def forward(self, reference: TensorDict, other: TensorDict) -> torch.Tensor:
        """
        Computes the specified metric between two TensorDicts.

        Args:
            reference (TensorDict): The ground truth data.
            other (TensorDict): The predicted data.

        Returns:
            torch.Tensor:
            - If `variable_mode='single'`, returns shape [B, T].
            - If `variable_mode='all'`, returns shape [B, T, C], where C
              is the number of variables (channels).
        """

        reference = reference.to(self.device)
        other = other.to(self.device)
        if self.variable_mode == "single":
            if (
                self.variable_name not in reference.keys()
                or self.variable_name not in other.keys()
            ):
                raise ValueError(
                    f"Specified variable_name '{self.variable_name}' not found in TensorDicts."
                )
            keys_to_process = [self.variable_name]
        else:  # 'all'
            # Process all keys present in both dicts, excluding metadata
            excluded_keys = {"seq_length", "cond_target", "cond_input", "obstacle_mask"}
            keys_to_process = [
                k
                for k in reference.keys()
                if k in other.keys() and k not in excluded_keys
            ]
            if not keys_to_process:
                logger.info(
                    "Warning: No common, non-excluded variables found to compare."
                )
                return torch.empty(0)

        per_var_results: List[torch.Tensor] = []
        for var in keys_to_process:
            ref_tensor = reference.get(var)
            other_tensor = other.get(var)

            # --- Input Validation ---
            if not isinstance(ref_tensor, torch.Tensor) or not isinstance(
                other_tensor, torch.Tensor
            ):
                logger.info(f"Skipping variable '{var}': not a tensor.")
                continue
            if ref_tensor.shape != other_tensor.shape:
                logger.info(f"Skipping variable '{var}': shapes mismatch.")
                logger.info(f"ref_tensor.shape: {ref_tensor.shape}")
                logger.info(f"other_tensor.shape: {other_tensor.shape}")
                continue
            # Ensure tensor is 4D [B, T, H, W]
            if ref_tensor.ndim != 4:
                logger.info(
                    f"Skipping variable '{var}': not an image-like 4D tensor (expected [B, T, H, W])."
                )
                continue

            # --- Dispatch to correct computation function ---
            if self.mode in self.LSIM_MODES:
                dist = self._compute_lsim_distance(ref_tensor, other_tensor)
            else:
                dist = self._compute_skimage_distance(ref_tensor, other_tensor)

            per_var_results.append(dist)

        if not per_var_results:
            return torch.empty(0)

        # --- CHANGE: Stack results to match user's snippet logic ---
        if self.variable_mode == "single":
            return per_var_results[0]  # Returns [B, T]
        else:
            # Stacks list of [B, T] tensors into [B, T, C]
            return torch.stack(per_var_results, dim=2).mean(dim=-1)

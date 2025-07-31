# models/metrics.py
import torch
import torch.nn as nn
import numpy as np
import skimage.metrics as sk_metrics
from tensordict import TensorDict
from typing import Optional, List
from pathlib import Path
import logging

from .lsim.distance_model import DistanceModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class Metric(nn.Module):
    """
    Computes image-based comparison metrics between two sets of video data
    represented as TensorDicts.

    This module supports standard image metrics (L2, SSIM, PSNR, VI) via
    scikit-image and learned perceptual metrics (LSIM_BASE) using
    the official DistanceModel wrapper.

    Args:
        mode (str): The metric to compute. Valid modes are "L2", "SSIM",
                    "PSNR", "VI", "LSIM_BASE".
        variable_mode (str): How to select variables from the TensorDict.
                             Must be 'single' or 'all'. Defaults to "single".
        variable_name (Optional[str]): The name of the variable to compute the
                                       metric on if `variable_mode` is 'single'.
                                       Defaults to None.
        lsim_model_path (Optional[str | Path]): Path to the pretrained LSIM
                                                model weights. Required if
                                                `mode` is "LSIM" or
                                                "". Defaults to
                                                "./models/LSIM/Models/LSiM.pth".

    Raises:
        ValueError: If an invalid `mode` or `variable_mode` is provided, or if
                    required arguments are missing.
    """

    VALID_MODES = ["L2", "SSIM", "PSNR", "VI", "LSIM"]
    SKIMAGE_MODES = ["L2", "SSIM", "PSNR", "VI"]
    LSIM_MODES = ["LSIM"]

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
        if mode in self.LSIM_MODES and not lsim_model_path:
            raise ValueError("`lsim_model_path` must be provided for LSIM modes")

        self.mode = mode
        self.variable_mode = variable_mode
        self.variable_name = variable_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Setup LSIM Model if required ---
        if self.mode in self.LSIM_MODES:
            if not lsim_model_path:
                raise ValueError("`lsim_model_path` must be provided for LSIM modes")
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
        logger.info("LSIM DistanceModel loaded successfully.")

    def _compute_lsim_distance(
        self, ref: torch.Tensor, other: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the LSIM distance using the DistanceModel wrapper.
        """
        assert ref.ndim == 4, f"Input tensors must be 4D, but got shape {ref.shape}"
        B, T, H, W = ref.shape
        distances_list = []

        for i in range(B):
            batch_distances = []
            for j in range(T):
                # Convert the single frame to a numpy array
                ref_np_2d = (ref[i, j].clamp(0, 1) * 255).byte().cpu().numpy()
                other_np_2d = (other[i, j].clamp(0, 1) * 255).byte().cpu().numpy()

                # The LSIM library's internal transform expects a 4D array: (B, H, W, C)
                # First, add the channel dimension to make it 3D
                ref_np_3d = ref_np_2d[..., np.newaxis]
                other_np_3d = other_np_2d[..., np.newaxis]

                # Then, add a batch dimension of 1
                ref_np_4d = np.expand_dims(ref_np_3d, axis=0)
                other_np_4d = np.expand_dims(other_np_3d, axis=0)

                # The computeDistance method now receives the correctly shaped array
                dist_val = self.lsim_model.computeDistance(ref_np_4d, other_np_4d)

                # Append the result as a standard Python float
                batch_distances.append(float(dist_val))
            distances_list.append(batch_distances)

        # Convert the list of lists to a single tensor at the end
        return torch.tensor(distances_list, dtype=torch.float32, device=ref.device)

    def _compute_skimage_distance(
        self, ref: torch.Tensor, other: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes scikit-image metrics between two 4D tensors [B, T, H, W].
        Assumes input tensors are normalized in the [0, 1] range.
        """
        assert ref.ndim == 4, f"Input tensors must be 4D, but got shape {ref.shape}"
        B, T, H, W = ref.shape

        # Move to CPU and convert to numpy once for efficiency
        ref_np = ref.cpu().numpy()
        other_np = other.cpu().numpy()

        distances = np.zeros((B, T), dtype=np.float32)

        for i in range(B):
            for j in range(T):
                r, o = ref_np[i, j], other_np[i, j]

                if self.mode == "L2":
                    # MSE on [0, 1] range is already normalized
                    distances[i, j] = sk_metrics.mean_squared_error(r, o)
                elif self.mode == "SSIM":
                    # SSIM is a similarity [0, 1], so 1 - SSIM is a distance
                    distances[i, j] = 1.0 - sk_metrics.structural_similarity(
                        r, o, data_range=1.0
                    )
                elif self.mode == "PSNR":
                    # PSNR is inverted to act as a distance (lower is better)
                    # A small epsilon prevents division by zero for identical images
                    mse = sk_metrics.mean_squared_error(r, o)
                    psnr = 20 * np.log10(1.0) - 10 * np.log10(mse + 1e-9)
                    distances[i, j] = -psnr
                elif self.mode == "VI":
                    # VI requires integer inputs
                    r_int = (r * 255).astype(np.uint8)
                    o_int = (o * 255).astype(np.uint8)
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
            torch.Tensor: A tensor of shape [B, T] with the computed distances.
                          If `variable_mode` is 'all', the distances are
                          averaged across all valid variables.
        """
        if self.variable_mode == "single":
            keys_to_process = [self.variable_name]
        else:  # 'all'
            # Process all keys present in both dicts, excluding metadata
            excluded_keys = {"seq_length", "Re_target", "Re_input", "obstacle_mask"}
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
                continue
            # Ensure tensor is 4D [B, T, H, W], assuming grayscale if 3D [B, T, D]
            if ref_tensor.ndim == 3:
                # Assuming 3D is [B, T, Features], which is not image-like. Skip.
                logger.info(f"Skipping variable '{var}': not an image-like 4D tensor.")
                continue
            if ref_tensor.ndim != 4:
                logger.info(f"Skipping variable '{var}': not an image-like 4D tensor.")
                continue

            # --- Dispatch to correct computation function ---
            if self.mode in self.LSIM_MODES:
                dist = self._compute_lsim_distance(ref_tensor, other_tensor)
            else:
                dist = self._compute_skimage_distance(ref_tensor, other_tensor)

            per_var_results.append(dist)

        if not per_var_results:
            return torch.empty(0)

        # Stack and average across variables if in 'all' mode
        # If 'single' mode, this just returns the single result
        return torch.stack(per_var_results, dim=0).mean(dim=0)

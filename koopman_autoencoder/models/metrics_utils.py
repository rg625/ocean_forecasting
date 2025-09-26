# models/metrics_utils.py

import numpy as np
import torch
from tensordict import TensorDict, stack as stack_tensordict
from typing import Protocol, Dict, Tuple, List, Optional, Any, cast
import logging
from models.dataloader import QGDatasetBase
from models.utils import cuda_timer, elapsed_time

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Diffusion model-specific normalization constants
DIFFUSION_MEAN = {"v_x": 0.444969, "v_y": 0.000299, "p": 0.000586, "rey": 550.0}
DIFFUSION_STD = {"v_x": 0.206128, "v_y": 0.206128, "p": 0.003942, "rey": 262.678467}


# =====================================================================================
# SECTION 1: CORE METRIC & DERIVED VARIABLE COMPUTATION
# =====================================================================================


class RolloutFn(Protocol):
    def __call__(
        self,
        model: Any,
        input_seq: TensorDict,
        metadata: dict,
        rollout_steps: int,
        dataset: Any = None,
    ) -> TensorDict: ...


def compute_vorticity(
    v_x: torch.Tensor, v_y: torch.Tensor, chunk_size: int = 16
) -> torch.Tensor:
    """
    Computes vorticity (dv_y/dx - dv_x/dy) from velocity fields in chunks to save memory.
    Expects input shape like [B, T, H, W] or [T, H, W].
    """
    vort_list = []
    # Add a temporary batch dimension if the input is a single sample
    is_single_sample = v_x.dim() == 3
    if is_single_sample:
        v_x, v_y = v_x.unsqueeze(0), v_y.unsqueeze(0)

    for i in range(0, v_x.size(0), chunk_size):
        vx_chunk = v_x[i : i + chunk_size]
        vy_chunk = v_y[i : i + chunk_size]

        # --- FIX ---
        # Calculate gradients along both spatial dimensions (H, W) at once.
        # torch.gradient returns gradients in the order of the dims provided.
        # For dim=(-2, -1), the output is (d/dy, d/dx).
        vx_dy, vx_dx = torch.gradient(vx_chunk, dim=(-2, -1))
        vy_dy, vy_dx = torch.gradient(vy_chunk, dim=(-2, -1))

        # Vorticity formula: (d(v_y)/dx - d(v_x)/dy)
        vort_list.append(vy_dx - vx_dy)

    result = torch.cat(vort_list, dim=0)

    # Remove the temporary batch dimension if it was added
    return result.squeeze(0) if is_single_sample else result


def compute_all_metrics(
    target: TensorDict,
    prediction: TensorDict,
    loader,
    variables: List[str],
    custom_min_max: Optional[Dict[str, Tuple[float, float]]] = None,
    chunk_size: int = 8,
) -> dict:
    """
    Computes a suite of metrics (L2, SSIM, PSNR) for specified variables.

    This function normalizes data to a [0, 1] range before metric calculation,
    as is standard for image-based metrics like SSIM and PSNR.
    """
    from models.metrics import Metric  # Local import to avoid circular dependency

    results: Dict = {mode: {} for mode in Metric.VALID_MODES}

    # --- Prepare Normalized Data ---
    target_norm, prediction_norm = {}, {}

    for var in variables:
        if custom_min_max and var in custom_min_max:
            vmin, vmax = custom_min_max[var]
            eps = 1e-8
            target_norm[var] = (target[var] - vmin) / (vmax - vmin + eps)
            prediction_norm[var] = (prediction[var] - vmin) / (vmax - vmin + eps)
        else:
            # Process in CPU chunks to avoid OOM with large datasets.
            target_chunks, pred_chunks = [], []
            for start in range(0, target.shape[0], chunk_size):
                end = start + chunk_size
                t_chunk = loader.to_unit_range(target[start:end].cpu())[var]
                p_chunk = loader.to_unit_range(prediction[start:end].cpu())[var]
                target_chunks.append(t_chunk)
                pred_chunks.append(p_chunk)
            target_norm[var] = torch.cat(target_chunks, dim=0)
            prediction_norm[var] = torch.cat(pred_chunks, dim=0)

    # Move all normalized data to the device for metric computation
    target_td = TensorDict(target_norm, batch_size=target.batch_size).to(DEVICE)
    pred_td = TensorDict(prediction_norm, batch_size=prediction.batch_size).to(DEVICE)

    # --- Calculate Metrics ---
    for mode in Metric.VALID_MODES:
        for var in variables + ["all"]:
            variable_mode = "all" if var == "all" else "single"
            metric_fn = Metric(
                mode=mode,
                variable_mode=variable_mode,
                variable_name=None if variable_mode == "all" else var,
            )
            dist = metric_fn(target_td, pred_td)  # Shape: [B, T]
            results[mode][var] = (dist.mean().item(), dist.std().item())

    return results


# =====================================================================================
# SECTION 2: DATA TRANSFORMATION UTILITIES (TENSOR <-> TENSORDICT)
# =====================================================================================


def tensor_to_tensordict(tensor: torch.Tensor, var_names: List[str]) -> TensorDict:
    """Converts a tensor [B, T, C, H, W] to a TensorDict."""
    num_samples, seq_len, channels, H, W = tensor.shape
    tensor_reshaped = tensor.permute(0, 1, 3, 4, 2)
    td_fields = {var: tensor_reshaped[..., i] for i, var in enumerate(var_names)}
    return TensorDict(td_fields, batch_size=[num_samples, seq_len])


def tensordict_to_tensor(
    td: TensorDict, var_names: List[str], re_val: Optional[float] = None
) -> torch.Tensor:
    """
    Converts a TensorDict to a tensor, optionally adding Reynolds number as a channel.
    Handles both batched (4D fields) and non-batched (3D fields) inputs.
    """
    first_key = var_names[0]
    field_shape = td[first_key].shape

    # Check dimensions and add a batch dimension if it's missing
    if len(field_shape) == 3:  # Input is a single sample with shape [T, H, W]
        td = td.unsqueeze(0)  # Add a batch dimension -> [1, T, H, W]
    elif len(field_shape) != 4:  # Input is not the expected [B, T, H, W]
        raise ValueError(
            f"Unexpected field shape in TensorDict: {field_shape}. "
            "Expected 3 dimensions [T, H, W] or 4 dimensions [B, T, H, W]."
        )

    # Now we can safely unpack the 4D shape
    B, T, H, W = td[first_key].shape

    stacked = torch.stack([td[var] for var in var_names], dim=-1)  # [B, T, H, W, C]

    if re_val is not None:
        re_tensor = torch.full(
            (B, T, H, W, 1), re_val, device=stacked.device, dtype=stacked.dtype
        )
        stacked = torch.cat([stacked, re_tensor], dim=-1)

    return stacked.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]


# =====================================================================================
# SECTION 3: DIFFUSION MODEL-SPECIFIC NORMALIZATION HELPERS
# =====================================================================================


def normalize_for_diffusion(td: TensorDict) -> TensorDict:
    """Applies Z-score normalization for the diffusion model."""
    normalized_td = td.clone()
    for key, mean in DIFFUSION_MEAN.items():
        if key in td.keys():
            normalized_td[key] = (td[key] - mean) / DIFFUSION_STD[key]
    return normalized_td


def denormalize_from_diffusion(td: TensorDict) -> TensorDict:
    """Reverses the diffusion model's Z-score normalization."""
    denormalized_td = td.clone()
    for key, mean in DIFFUSION_MEAN.items():
        if key in td.keys():
            denormalized_td[key] = (td[key] * DIFFUSION_STD[key]) + mean
    return denormalized_td


# =====================================================================================
# SECTION 4: MODEL-SPECIFIC ROLLOUT FUNCTIONS
# =====================================================================================


def ke_timeseries(tensordict, dx=1.0, dy=1.0, rho=1.0):
    """
    Compute total kinetic energy for each time step.

    Args:
        tensordict: TensorDict with keys "v_x", "v_y"
        dx, dy: grid spacing in x and y
        rho: fluid density

    Returns:
        Tensor of shape [61] with KE at each time step
    """
    vx = tensordict["v_x"]  # shape [61, 64, 128]
    vy = tensordict["v_y"]  # shape [61, 64, 128]

    # kinetic energy density (per cell, per timestep)
    ke_density = 0.5 * (vx**2 + vy**2)  # [61, 64, 128]

    # integrate over space: sum over spatial dimensions
    ke_total = rho * torch.sum(ke_density, dim=(1, 2)) * dx * dy  # [61]

    return ke_total

    # vx = torch.mean(tensordict["v_x"], dim=(1, 2))  # shape [T, 64, 128]
    # vy = torch.mean(tensordict["v_y"], dim=(1, 2))  # shape [T, 64, 128]

    # # kinetic energy density (per cell, per timestep)
    # ke_density = 0.5 * (vx**2 + vy**2)  # [T, 64, 128]
    # return rho * ke_density * dx * dy  # [61]


def run_kae_rollout(
    model,
    input_seq: TensorDict,
    rollout_steps: int,
    return_xpreds: Optional[bool] = True,
) -> TensorDict:
    """Performs a long rollout for a Koopman Autoencoder (KAE) model."""
    input_seq = input_seq.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # The model returns a TensorDict containing the full predicted sequence
        predicted_td = model(input_seq, seq_length=rollout_steps)
    return (
        predicted_td.x_preds.squeeze(0) if return_xpreds else predicted_td
    )  # Remove batch dim


def kae_rollout_wrapper(
    model, input_seq, metadata: dict, rollout_steps: int, dataset=None
):
    """
    Wraps run_kae_rollout to match the signature of run_diffusion_rollout.
    The `metadata` and `dataset` arguments are ignored for KAE.
    """
    return run_kae_rollout(
        model=model, input_seq=input_seq, rollout_steps=rollout_steps
    )


def run_diffusion_rollout(
    model, input_seq: TensorDict, metadata: Dict, rollout_steps: int, dataset
) -> TensorDict:
    """Performs a long rollout for the Diffusion model, handling its unique data requirements.
    Returns predictions and a timings dict with total, average, and per-step timings.
    """
    model.eval()
    timings: Dict = {}
    var_names_3c = ["v_x", "v_y", "p"]

    total_start, total_end = cuda_timer()
    total_start.record()

    with torch.no_grad():
        # Step 1: Denormalize from dataset format, then re-normalize for diffusion model
        input_denorm = dataset.denormalize(input_seq.clone())
        input_renorm = normalize_for_diffusion(input_denorm)

        # Step 2: Get and normalize the Reynolds number for conditioning
        re_val = metadata["Re_target"][0]

        # Ensure it's a tensor
        if not torch.is_tensor(re_val):
            re_val = torch.tensor([re_val], dtype=torch.float32, device=DEVICE)

        # If empty, replace with a constant (e.g., dataset mean) to avoid crash
        if re_val.numel() == 0:
            re_val = torch.tensor(
                [DIFFUSION_MEAN["rey"]], dtype=torch.float32, device=DEVICE
            )

        # Normalize
        normalized_re = (re_val - DIFFUSION_MEAN["rey"]) / DIFFUSION_STD["rey"]

        # If length < rollout_steps, pad by repeating last value
        if normalized_re.numel() < rollout_steps:
            normalized_re = normalized_re.repeat(rollout_steps)

        re_input = (
            metadata["Re_input"][0].item()
            if metadata["Re_input"][0].ndim == 0
            else metadata["Re_input"][0][-1]
        )
        normalized_re_input = (re_input - DIFFUSION_MEAN["rey"]) / DIFFUSION_STD["rey"]

        # Step 3: Convert to a 4-channel tensor [vx, vy, p, Re] and transpose H, W
        d = tensordict_to_tensor(
            input_renorm, var_names_3c, re_val=normalized_re_input
        ).to(DEVICE)
        d = d.permute(0, 1, 2, 4, 3)  # [B, T, C, H, W] -> [B, T, C, W, H]

        # Step 4: Autoregressive prediction loop
        B, T_in, C, H, W = d.shape
        T_out = T_in + rollout_steps
        prediction = torch.zeros([B, T_out, C, H, W], device=DEVICE)
        prediction[:, :T_in] = d

        step_times = []
        for i in range(T_in, T_out):
            start, end = cuda_timer()
            start.record()
            torch.cuda.synchronize()

            # Conditioning with previous 2 steps
            cond = torch.cat(
                [prediction[:, i - 2 : i - 1], prediction[:, i - 1 : i]], dim=2
            )
            current_slice = prediction[:, i - 1 : i]
            result = model(conditioning=cond, data=current_slice)

            end.record()
            torch.cuda.synchronize()
            step_times.append(elapsed_time(start, end))

            # Overwrite predicted Reynolds number with constant
            result[..., -1, :, :] = normalized_re[i - T_in]
            prediction[:, i : i + 1] = result

    # Step 5: Convert back to TensorDict and denormalize
    output_tensor = prediction[:, T_in:].permute(0, 1, 2, 4, 3)  # Transpose back H, W
    var_names_4c = var_names_3c + ["rey"]
    pred_td_normalized = tensor_to_tensordict(output_tensor.cpu(), var_names_4c)
    pred_td_denorm = denormalize_from_diffusion(pred_td_normalized)
    pred_td_denorm.pop("rey", None)  # Remove temporary Reynolds number field

    # Final total timing
    total_end.record()
    torch.cuda.synchronize()
    total_time = elapsed_time(total_start, total_end)

    # Organize timings in the requested order
    timings = {
        "total_time_ms": float(total_time),
        "average_step_time_ms": float(np.mean(step_times)),
        "all_step_times_ms": [float(t) for t in step_times],
    }
    logger.info(f"Diffusion timings: {timings}")

    return cast(TensorDict, pred_td_denorm.squeeze(0))


# =====================================================================================
# SECTION 5: UNIFIED EVALUATION ORCHESTRATION
# =====================================================================================


def generate_predictions_for_dataset(
    model: torch.nn.Module,
    dataset: QGDatasetBase,
    input_len: int,
    output_len: int,
    rollout_fn: RolloutFn,
) -> Tuple[TensorDict, TensorDict]:
    """
    Iterates through a dataset to generate model predictions for all samples.
    """
    all_targets, all_predictions = [], []

    for idx in range(0, len(dataset), (output_len + input_len)):
        try:
            input_seq, ground_truth_future, metadata = dataset[idx]

            # Add required metadata to the input TensorDict for the model.
            # This is necessary for models that use conditioning (e.g., on Re or obstacles).
            if "Re_input" in metadata:
                # The .repeat() call assumes the metadata is for a single sample and adds a
                # compatible batch dimension if the input_seq has one.
                input_seq["Re_input"] = metadata["Re_input"][0].repeat(
                    *input_seq.batch_size
                )
            if "obstacle_mask" in metadata:
                input_seq["obstacle_mask"] = metadata["obstacle_mask"][0].repeat(
                    *input_seq.batch_size, 1, 1
                )
            if rollout_fn is None:
                raise ValueError("rollout_fn must be provided")
            predicted_future = rollout_fn(
                model, input_seq, metadata, output_len, dataset
            )

            denormalized_gt = dataset.denormalize(ground_truth_future.clone())

            if "diffusion" in rollout_fn.__name__:
                denormalized_pred = predicted_future
            else:
                denormalized_pred = dataset.denormalize(predicted_future.clone())

            all_targets.append(denormalized_gt)
            all_predictions.append(denormalized_pred)

        except Exception as e:
            logger.error(f"Error processing sample at index {idx}: {e}", exc_info=True)
            continue

    if not all_predictions:
        logger.error("No valid predictions were generated. Aborting evaluation.")
        return TensorDict({}, batch_size=[0]), TensorDict({}, batch_size=[0])

    stacked_targets = stack_tensordict(all_targets, dim=0)
    stacked_predictions = stack_tensordict(all_predictions, dim=0)

    logger.info(f"Generated predictions for {stacked_targets.shape[0]} samples.")
    return stacked_targets, stacked_predictions


def run_evaluation(
    model: torch.nn.Module,
    loader,
    input_len: int,
    output_len: int,
    rollout_fn: RolloutFn,
) -> Dict:
    """
    Orchestrates the entire evaluation pipeline for a given model.
    1. Generates predictions for the full dataset.
    2. Computes derived quantities (e.g., vorticity).
    3. Calculates and returns all metrics.
    """
    rollout_name = rollout_fn.__name__ if rollout_fn is not None else "<undefined>"
    logger.info(
        f"Starting evaluation for model '{model.__class__.__name__}' using '{rollout_name}'..."
    )

    # --- 1. Generate Predictions ---
    targets, predictions = generate_predictions_for_dataset(
        model=model,
        dataset=loader.dataset,
        input_len=input_len,
        output_len=output_len,
        rollout_fn=rollout_fn,
    )

    if targets.is_empty():
        logger.error(
            "Prediction generation failed. Cannot proceed with metric calculation."
        )
        return {}

    # --- 2. Compute Derived Variables (e.g., Vorticity) ---
    logger.info("Computing derived variables...")
    custom_min_max = {}
    if "v_x" in targets and "v_y" in targets:
        vort_truth = compute_vorticity(targets["v_x"], targets["v_y"])
        targets["vort"] = vort_truth

        vort_pred = compute_vorticity(predictions["v_x"], predictions["v_y"])
        predictions["vort"] = vort_pred

        # Use the ground truth vorticity to define the normalization range
        vmin, vmax = vort_truth.min().item(), vort_truth.max().item()
        custom_min_max["vort"] = (vmin, vmax)
        logger.info(
            f"Global vorticity range for normalization: [{vmin:.4f}, {vmax:.4f}]"
        )
    else:
        logger.warning("Velocity fields 'v_x' and 'v_y' not found. Skipping vorticity.")

    # --- 3. Compute Metrics ---
    vars_to_eval = [k for k in predictions.keys() if k in targets]
    logger.info(f"Computing metrics for variables: {vars_to_eval}")

    metrics = compute_all_metrics(
        target=targets,
        prediction=predictions,
        loader=loader,
        variables=vars_to_eval,
        custom_min_max=custom_min_max,
    )

    logger.info("Evaluation complete.")
    return metrics

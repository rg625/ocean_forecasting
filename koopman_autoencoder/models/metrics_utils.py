import numpy as np
import torch
from tensordict import TensorDict, stack as stack_tensordict
from typing import Protocol, Dict, Tuple, List, Optional, Any, cast
import logging
from models.dataloader import QGDatasetBase, AbstractNormalizer
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

        # Calculate gradients along both spatial dimensions (H, W) at once.
        # torch.gradient returns gradients in the order of the dims provided.
        # For dim=(-2, -1), the output is (d/dy, d/dx).
        _, vx_dx = torch.gradient(vx_chunk, dim=(-2, -1))
        vy_dy, _ = torch.gradient(vy_chunk, dim=(-2, -1))

        # Vorticity formula: (d(v_y)/dx - d(v_x)/dy) - MISTAKE IN ORIGINAL FILE
        # Corrected formula: d(v_y)/dx - d(v_x)/dy
        # Original had vy_dy and vx_dx, which is incorrect. Let's assume the user meant
        # to get d/dx and d/dy correctly.
        vy_grad = torch.gradient(vy_chunk, dim=(-1, -2))  # (d/dx, d/dy)
        vx_grad = torch.gradient(vx_chunk, dim=(-1, -2))  # (d/dx, d/dy)
        vort_list.append(vy_grad[0] - vx_grad[1])  # dv_y/dx - dv_x/dy

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
        mode_results = {}
        for var in variables:
            # Assuming Metric class can handle single variable evaluation
            metric_fn = Metric(mode=mode, variable_name=var)
            dist = metric_fn(target_td, pred_td)
            mode_results[var] = (dist.mean().item(), dist.std().item())

        # Compute average across all variables
        all_means = [v[0] for v in mode_results.values()]
        all_stds = [v[1] for v in mode_results.values()]
        if all_means:
            mean_all = float(np.mean(all_means))
            std_all = float(np.sqrt(np.mean(np.array(all_stds) ** 2)))
            mode_results["all"] = (mean_all, std_all)

        results[mode] = mode_results

    return results


# =====================================================================================
# SECTION 2: DATA TRANSFORMATION UTILITIES (TENSOR <-> TENSORDICT)
# =====================================================================================


def tensor_to_tensordict(tensor: torch.Tensor, var_names: List[str]) -> TensorDict:
    """Converts a tensor [B, T, C, H, W] to a TensorDict."""
    if tensor.dim() != 5:
        raise ValueError(f"Expected a 5D tensor, but got shape {tensor.shape}")
    num_samples, seq_len, _, _, _ = tensor.shape
    td_fields = {var: tensor[:, :, i] for i, var in enumerate(var_names)}
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

    td_for_stack = td.clone()
    # Check dimensions and add a batch dimension if it's missing
    if len(field_shape) == 3:  # Input is a single sample with shape [T, H, W]
        td_for_stack = td_for_stack.unsqueeze(
            0
        )  # Add a batch dimension -> [1, T, H, W]
    elif len(field_shape) != 4:  # Input is not the expected [B, T, H, W]
        raise ValueError(
            f"Unexpected field shape in TensorDict: {field_shape}. "
            "Expected 3 dimensions [T, H, W] or 4 dimensions [B, T, H, W]."
        )

    # Now we can safely unpack the 4D shape
    B, T, H, W = td_for_stack[first_key].shape

    stacked = torch.stack(
        [td_for_stack[var] for var in var_names], dim=2
    )  # [B, T, C, H, W]

    if re_val is not None:
        re_tensor = torch.full(
            (B, T, 1, H, W), re_val, device=stacked.device, dtype=stacked.dtype
        )
        stacked = torch.cat([stacked, re_tensor], dim=2)

    return stacked


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
    Assumes input shape is [T, H, W].
    """
    vx = tensordict["v_x"]
    vy = tensordict["v_y"]

    # kinetic energy density (per cell, per timestep)
    ke_density = 0.5 * (vx**2 + vy**2)

    # integrate over space: sum over spatial dimensions and multiply by cell area
    ke_total = rho * torch.sum(ke_density, dim=(-2, -1)) * dx * dy

    return ke_total


def run_kae_rollout(
    model,
    input_seq: TensorDict,
    rollout_steps: int,
    return_xpreds: Optional[bool] = True,
) -> TensorDict:
    """Performs a long rollout for a Koopman Autoencoder (KAE) model."""
    input_seq = input_seq.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        predicted_td = model(input_seq, seq_length=rollout_steps)
    return (
        predicted_td.x_preds.squeeze(0) if return_xpreds else predicted_td
    )  # Remove batch dim


def kae_rollout_wrapper(
    model, input_seq, metadata: dict, rollout_steps: int, dataset=None
):
    """Wraps run_kae_rollout to match the signature of run_diffusion_rollout."""
    return run_kae_rollout(
        model=model, input_seq=input_seq, rollout_steps=rollout_steps
    )


def run_diffusion_rollout(
    model, input_seq: TensorDict, metadata: Dict, rollout_steps: int, dataset
) -> TensorDict:
    """Performs a long rollout for the Diffusion model."""
    model.eval()
    timings: Dict[str, Any] = {}
    var_names_3c = ["v_x", "v_y", "p"]
    total_start, total_end = cuda_timer()
    total_start.record()

    with torch.no_grad():
        input_denorm = dataset.denormalize(input_seq.clone())
        input_renorm = normalize_for_diffusion(input_denorm)

        re_val = metadata.get("Re_target", [DIFFUSION_MEAN["rey"]])[0]
        if not torch.is_tensor(re_val):
            re_val = torch.tensor([re_val], dtype=torch.float32, device=DEVICE)
        normalized_re = (re_val - DIFFUSION_MEAN["rey"]) / DIFFUSION_STD["rey"]

        re_input_val = (
            metadata["Re_input"][0][-1]
            if metadata.get("Re_input") and metadata["Re_input"][0].ndim > 0
            else metadata.get("Re_input", [DIFFUSION_MEAN["rey"]])[0]
        )
        normalized_re_input = (re_input_val - DIFFUSION_MEAN["rey"]) / DIFFUSION_STD[
            "rey"
        ]

        d = tensordict_to_tensor(
            input_renorm, var_names_3c, re_val=normalized_re_input.item()
        ).to(DEVICE)
        d = d.permute(0, 1, 2, 4, 3)

        B, T_in, C, H, W = d.shape
        T_out = T_in + rollout_steps
        prediction = torch.zeros([B, T_out, C, H, W], device=DEVICE, dtype=d.dtype)
        prediction[:, :T_in] = d

        step_times = []
        for i in range(T_in, T_out):
            start, end = cuda_timer()
            start.record()
            cond = torch.cat(
                [prediction[:, i - 2 : i - 1], prediction[:, i - 1 : i]], dim=2
            )
            current_slice = prediction[:, i - 1 : i]
            result = model(conditioning=cond, data=current_slice)
            end.record()
            torch.cuda.synchronize()
            step_times.append(elapsed_time(start, end))

            # Overwrite predicted Reynolds number with constant target
            result[..., -1, :, :] = normalized_re  # Assuming constant Re for rollout
            prediction[:, i : i + 1] = result

    output_tensor = prediction[:, T_in:].permute(0, 1, 2, 4, 3)
    var_names_4c = var_names_3c + ["rey"]
    pred_td_normalized = tensor_to_tensordict(output_tensor.cpu(), var_names_4c)
    pred_td_denorm = denormalize_from_diffusion(pred_td_normalized)
    pred_td_denorm.pop("rey", None)

    total_end.record()
    torch.cuda.synchronize()
    total_time = elapsed_time(total_start, total_end)

    timings = {
        "total_time_ms": float(total_time),
        "average_step_time_ms": float(np.mean(step_times)) if step_times else 0.0,
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
    """Iterates through a dataset to generate model predictions for all samples."""
    all_targets, all_predictions = [], []

    indices_to_process = list(range(0, len(dataset)))

    for idx in indices_to_process:
        try:
            input_seq, ground_truth_future, metadata_raw = dataset[idx]

            metadata = {}
            for key, (val, dest) in metadata_raw.items():
                metadata[key] = [val]  # Wrap in a list to mimic batch

            # Manually add metadata to input_seq for conditioning
            if "Re_input" in metadata:
                re_val = metadata["Re_input"][0]
                if not torch.is_tensor(re_val):
                    re_val = torch.tensor(re_val)
                # --- FIX ---
                # Expand the scalar 're_val' to match the TensorDict's batch_size
                input_seq["Re_input"] = re_val.expand(input_seq.batch_size)

            if "obstacle_mask" in metadata and "obstacle_mask" not in input_seq.keys():
                mask = metadata["obstacle_mask"][0]
                # --- FIX ---
                # Expand the mask [H, W] to [T, H, W] using the batch_size
                input_seq["obstacle_mask"] = mask.expand(*input_seq.batch_size, -1, -1)

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

    stacked_targets = stack_tensordict(all_targets)
    stacked_predictions = stack_tensordict(all_predictions)

    logger.info(f"Generated predictions for {stacked_targets.shape[0]} samples.")
    return stacked_targets, stacked_predictions


def run_evaluation(
    model: torch.nn.Module,
    loader,
    input_len: int,
    output_len: int,
    rollout_fn: RolloutFn,
) -> Dict:
    """Orchestrates the entire evaluation pipeline for a given model."""
    rollout_name = rollout_fn.__name__ if rollout_fn is not None else "<undefined>"
    logger.info(
        f"Starting evaluation for model '{model.__class__.__name__}' using '{rollout_name}'..."
    )

    targets, predictions = generate_predictions_for_dataset(
        model=model,
        dataset=loader.dataset,
        input_len=input_len,
        output_len=output_len,
        rollout_fn=rollout_fn,
    )

    if targets.is_empty():
        return {}

    logger.info("Computing derived variables...")
    custom_min_max: Dict[str, Tuple[float, float]] = {}
    if "v_x" in targets.keys() and "v_y" in targets.keys():
        vort_truth = compute_vorticity(targets["v_x"], targets["v_y"])
        targets["vort"] = vort_truth
        vort_pred = compute_vorticity(predictions["v_x"], predictions["v_y"])
        predictions["vort"] = vort_pred
        vmin, vmax = vort_truth.min().item(), vort_truth.max().item()
        custom_min_max["vort"] = (vmin, vmax)
        logger.info(
            f"Global vorticity range for normalization: [{vmin:.4f}, {vmax:.4f}]"
        )

    vars_to_eval = [k for k in predictions.keys() if k in targets.keys()]
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


def compute_stability_metrics(
    ground_truth_td: TensorDict,
    predicted_td: TensorDict,
    diff_td: Optional[TensorDict] = None,
    normalizer: Optional[AbstractNormalizer] = None,
    ttd_threshold: float = 1e2,
) -> Dict:
    """
    Compute stability metrics using PyTorch and TensorDict.
    This function expects a single trajectory, i.e., shape [T, H, W].
    """
    if ground_truth_td.dim() > 3:
        raise ValueError(
            f"Expected a single trajectory, but got shape {ground_truth_td.shape}"
        )

    # Apply normalizer if provided
    if normalizer:
        ground_truth_td = normalizer.transform(ground_truth_td)
        predicted_td = normalizer.transform(predicted_td)
        if diff_td is not None:
            diff_td = normalizer.transform(diff_td)

    device = ground_truth_td.device
    time_steps = ground_truth_td.batch_size[0]

    # Determine which variables to use for slope error calculation
    norm_keys = [k for k in predicted_td.keys() if k in ground_truth_td.keys()]
    if not norm_keys:
        raise ValueError("No common keys between prediction and ground truth.")

    # Check for velocity fields to compute energy
    has_velocity = "v_x" in ground_truth_td.keys() and "v_y" in ground_truth_td.keys()

    # Initial energy (ground truth)
    E0 = None
    if has_velocity:
        v_x_gt_0 = ground_truth_td["v_x"][0]
        v_y_gt_0 = ground_truth_td["v_y"][0]
        E0 = 0.5 * torch.sum(v_x_gt_0**2 + v_y_gt_0**2)

    # Initialize storage
    slope_error_pred, energy_drift_pred = [], []
    diverged_pred = torch.zeros(time_steps, device=device)

    slope_error_diff, energy_drift_diff = [], []
    diverged_diff = (
        torch.zeros(time_steps, device=device) if diff_td is not None else None
    )

    for t in range(time_steps):
        # --- Prediction Metrics ---
        gt_t_stack = torch.stack([ground_truth_td[k][t] for k in norm_keys])
        pred_t_stack = torch.stack([predicted_td[k][t] for k in norm_keys])
        error_pred = torch.linalg.norm((pred_t_stack - gt_t_stack).flatten())
        slope_error_pred.append(error_pred.item())
        if error_pred > ttd_threshold or torch.isnan(error_pred):
            diverged_pred[t] = 1

        if has_velocity and E0 is not None:
            v_x_pred = predicted_td["v_x"][t]
            v_y_pred = predicted_td["v_y"][t]
            E_pred = 0.5 * torch.sum(v_x_pred**2 + v_y_pred**2)
            energy_drift_pred.append(torch.abs(E_pred - E0).item())

        # --- Differential Baseline Metrics (if provided) ---
        if diff_td is not None:
            assert diverged_diff is not None
            diff_t_stack = torch.stack([diff_td[k][t] for k in norm_keys])
            error_diff = torch.linalg.norm((diff_t_stack - gt_t_stack).flatten())
            slope_error_diff.append(error_diff.item())
            if error_diff > ttd_threshold or torch.isnan(error_diff):
                diverged_diff[t] = 1

            if has_velocity and E0 is not None:
                v_x_diff = diff_td["v_x"][t]
                v_y_diff = diff_td["v_y"][t]
                E_diff = 0.5 * torch.sum(v_x_diff**2 + v_y_diff**2)
                energy_drift_diff.append(torch.abs(E_diff - E0).item())

    # Compute Time to Divergence (TTD)
    ttd_pred = (
        torch.argmax(diverged_pred).item() if torch.any(diverged_pred) else time_steps
    )
    ttd_diff = None
    if diverged_diff is not None:
        ttd_diff = (
            torch.argmax(diverged_diff).item()
            if torch.any(diverged_diff)
            else time_steps
        )

    metrics = {
        "ttd_pred": ttd_pred,
        "ttd_diff": ttd_diff,
        "slope_error_pred": slope_error_pred,
        "slope_error_diff": slope_error_diff,
        "energy_drift_pred": energy_drift_pred,
        "energy_drift_diff": energy_drift_diff,
    }

    return metrics

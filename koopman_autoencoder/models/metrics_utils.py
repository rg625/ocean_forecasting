import numpy as np
import torch
from tensordict import TensorDict, stack as stack_tensordict
from typing import (
    Protocol,
    Dict,
    Tuple,
    List,
    Optional,
    Any,
    cast,
    Generator,
)  # Added Generator
import logging
from .dataloader import QGDatasetBase, AbstractNormalizer
from .utils import cuda_timer, elapsed_time, save_timing_to_json
from .metrics import Metric
import os  # Added os for path operations
import random

seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Diffusion model-specific normalization constants
INC_MEAN = {"v_x": 0.444969, "v_y": 0.000299, "p": 0.000586, "Re": 550.0}
INC_STD = {"v_x": 0.206128, "v_y": 0.206128, "p": 0.003942, "Re": 262.678467}

QG_MEAN = {
    "q1": 6.130118365440265e-07,
    "q2": -6.069969143104177e-07,
    "forcing": 4.598683515047508e-12,
}
QG_STD = {
    "q1": 4.220613803016554e-05,
    "q2": 2.8499078149814454e-06,
    "forcing": 1.5010150481725185e-12,
}

TRA_MEAN = {
    "v_x": 0.560642,
    "v_y": -0.000129,
    "p": 0.637941,
    "rho": 0.903352,
    "Ma": 0.700000,
}
TRA_STD = {
    "v_x": 0.216987,
    "v_y": 0.216987,
    "p": 0.119944,
    "rho": 0.145391,
    "Ma": 0.118322,
}
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
        vy_grad = torch.gradient(vy_chunk, dim=(-1, -2))  # (d/dx, d/dy)
        vx_grad = torch.gradient(vx_chunk, dim=(-1, -2))  # (d/dx, d/dy)
        vort_list.append(vy_grad[0] - vx_grad[1])  # dv_y/dx - dv_x/dy

    result = torch.cat(vort_list, dim=0)

    # Remove the temporary batch dimension if it was added
    return result.squeeze(0) if is_single_sample else result


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
    td: TensorDict, var_names: List[str], cond_val: Optional[float] = None
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

    if cond_val is not None:
        re_tensor = torch.full(
            (B, T, 1, H, W), cond_val, device=stacked.device, dtype=stacked.dtype
        )
        stacked = torch.cat([stacked, re_tensor], dim=2)

    return stacked


# =====================================================================================
# SECTION 3: MODEL-SPECIFIC ROLLOUT FUNCTIONS
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


def ke_timeseries_spatial(tensordict, rho=1.0):
    """
    Compute per-pixel kinetic energy for each time step.
    Assumes input shape is [T, H, W].

    Returns:
        ke_pixel : [T, H, W] kinetic energy per pixel
    """
    vx = tensordict["v_x"]
    vy = tensordict["v_y"]

    # kinetic energy per pixel
    ke_pixel = 0.5 * rho * (vx**2 + vy**2)

    return ke_pixel


def run_kae_rollout(
    model,
    input_seq: TensorDict,
    rollout_steps: int,
    return_xpreds: Optional[bool] = True,
) -> TensorDict:
    """Performs a long rollout for a Koopman Autoencoder (KAE) model."""
    input_seq = input_seq.unsqueeze(0).to(
        DEVICE
    )  # if input_seq.ndim == 4 else input_seq
    # input_seq = input_seq.to(DEVICE)
    # print(f"input_seq: {input_seq['cond_input'][0]}")

    with torch.no_grad():
        predicted_td = model(input_seq, seq_length=rollout_steps)
        save_timing_to_json(
            timing_data=model.timings,
            model_name="kae_model",
            filename="kae_model_timings.json",
        )
    # logger.info(f"Saved Diffusion timings to: kae_model_timings.json")

    return (
        predicted_td.x_preds.squeeze(0) if return_xpreds else predicted_td
    )  # Remove batch dim


def run_exp_kae_rollout(
    model,
    input_seq: TensorDict,
    rollout_steps: int,
    return_xpreds: Optional[bool] = True,
) -> TensorDict:
    """Performs a long rollout for a Koopman Autoencoder (KAE) model."""
    input_seq = input_seq.unsqueeze(0).to(
        DEVICE
    )  # if input_seq.ndim == 4 else input_seq
    # input_seq = input_seq.to(DEVICE)
    # print(f"input_seq: {input_seq['cond_input'][0]}")

    with torch.no_grad():
        predicted_td = model.forward_theoretical(input_seq, seq_length=rollout_steps)
        save_timing_to_json(
            timing_data=model.timings,
            model_name="kae_model",
            filename="kae_model_timings_exp.json",
        )
    # logger.info(f"Saved Diffusion timings to: kae_model_timings.json")

    return (
        predicted_td.x_preds.squeeze(0) if return_xpreds else predicted_td
    )  # Remove batch dim


def kae_rollout_wrapper(
    model, input_seq, metadata: dict, rollout_steps: int, dataset=None
):
    """Wraps run_kae_rollout to match the signature of run_diffusion_rollout_inc."""
    return run_kae_rollout(
        model=model, input_seq=input_seq, rollout_steps=rollout_steps
    )


def run_diffusion_rollout(
    model,
    input_seq: TensorDict,
    metadata: Dict,
    rollout_steps: int,
    dataset,
    regime: str,  # "inc" or "tra"
) -> TensorDict:
    """Performs a long rollout for the Diffusion model."""

    assert regime in {"inc", "tra"}, f"Invalid regime: {regime}"

    model.eval()
    timings: Dict[str, Any] = {}

    # ----- Regime-specific configuration -----
    if regime == "inc":
        var_names = ["v_x", "v_y", "p"]
        MEAN = INC_MEAN
        STD = INC_STD
        cond_key = "Re"
        model_name = "diffusion_model_inc"
        timing_file = None
    else:  # "tra"
        var_names = ["v_x", "v_y", "p", "rho"]
        MEAN = TRA_MEAN
        STD = TRA_STD
        cond_key = "Ma"
        model_name = "diffusion_model_tra"
        timing_file = "diffusion_model_tra_timings.json"

    total_start, total_end = cuda_timer()
    total_start.record()

    with torch.no_grad():
        # ----- Conditioning target -----
        cond_val = metadata.get("cond_target", [MEAN[cond_key]])[0]
        if not torch.is_tensor(cond_val):
            cond_val = torch.tensor([cond_val], dtype=torch.float32, device=DEVICE)

        normalized_cond_target = (cond_val - MEAN[cond_key]) / STD[cond_key]

        # ----- Conditioning input -----
        cond_input_val = (
            metadata["cond_input"][0][-1]
            if metadata.get("cond_input") and metadata["cond_input"][0].ndim > 0
            else metadata.get("cond_input", [MEAN[cond_key]])[0]
        )

        normalized_cond_input = (cond_input_val - MEAN[cond_key]) / STD[cond_key]

        # ----- Input tensor -----
        d = tensordict_to_tensor(
            input_seq,
            var_names,
            cond_val=normalized_cond_input.item(),
        ).to(DEVICE)

        d = d.permute(0, 1, 2, 4, 3)

        B, T_in, C, H, W = d.shape
        T_out = T_in + rollout_steps

        prediction = torch.zeros(
            [B, T_out, C, H, W],
            device=DEVICE,
            dtype=d.dtype,
        )
        prediction[:, :T_in] = d

        step_times = []

        # ----- Rollout loop -----
        for i in range(T_in, T_out):
            start, end = cuda_timer()
            start.record()

            cond = torch.cat(
                [prediction[:, i - 2 : i - 1], prediction[:, i - 1 : i]],
                dim=2,
            )
            current_slice = prediction[:, i - 1 : i]
            result = model(conditioning=cond, data=current_slice)

            end.record()
            torch.cuda.synchronize()
            step_times.append(elapsed_time(start, end))

            # Overwrite conditioning channel with constant target
            result[..., -1, :, :] = normalized_cond_target
            prediction[:, i : i + 1] = result

    # ----- Output formatting -----
    output_tensor = prediction[:, T_in:].permute(0, 1, 2, 4, 3)
    var_names_out = var_names + [cond_key]

    pred_td_normalized = tensor_to_tensordict(
        output_tensor.cpu(),
        var_names_out,
    )
    pred_td_normalized.pop(cond_key, None)

    total_end.record()
    torch.cuda.synchronize()

    total_time = elapsed_time(total_start, total_end)

    timings = {
        "total_time_ms": float(total_time),
        "average_step_time_ms": float(np.mean(step_times)) if step_times else 0.0,
        "all_step_times_ms": [float(t) for t in step_times],
    }

    logger.info(f"Diffusion timings ({regime}): {timings}")

    if timing_file is not None:
        save_timing_to_json(
            timing_data=timings,
            model_name=model_name,
            filename=timing_file,
        )

    return cast(TensorDict, pred_td_normalized.squeeze(0))


# =====================================================================================
# SECTION 4: UNIFIED EVALUATION ORCHESTRATION
# =====================================================================================


def generate_predictions_for_dataset(
    model: torch.nn.Module,
    dataset: QGDatasetBase,
    output_len: int,
    rollout_fn: RolloutFn,
) -> Generator[Tuple[TensorDict, TensorDict, dict, int], None, None]:
    """
    Iterates through a dataset, generates model predictions,
    and yields each (target, prediction, metadata, idx) tuple.
    """
    indices_to_process = list(range(0, len(dataset)))

    for idx in indices_to_process:
        try:
            input_seq, ground_truth_future, metadata_raw = dataset[idx]
            if "KoopmanAutoencoder" in model.__class__.__name__:
                logger.info("Applying Mask for KAE")
                input_seq = dataset.apply_mask(input_seq)
                ground_truth_future = dataset.apply_mask(ground_truth_future)

            metadata = {}
            for key, (val, dest) in metadata_raw.items():
                metadata[key] = [val]  # Wrap in a list to mimic batch

            # Manually add metadata to input_seq for conditioning
            if "cond_input" in metadata:
                cond_val = metadata["cond_input"][0]
                if not torch.is_tensor(cond_val):
                    cond_val = torch.tensor(cond_val)

                # Expand the scalar 'cond_val' to match the TensorDict's batch_size
                input_seq["cond_input"] = cond_val.expand(input_seq.batch_size)

            if "obstacle_mask" in metadata and "obstacle_mask" not in input_seq.keys():
                mask = metadata["obstacle_mask"][0]
                # Expand the mask [H, W] to [T, H, W] using the batch_size
                input_seq["obstacle_mask"] = mask.expand(*input_seq.batch_size, -1, -1)

            predicted_future = rollout_fn(
                model, input_seq, metadata, output_len, dataset
            )

            denormalized_gt = dataset.denormalize(ground_truth_future.clone())
            denormalized_pred = dataset.denormalize(predicted_future.clone())

            # Yield the result for this sample
            yield denormalized_gt, denormalized_pred, metadata, idx

        except Exception as e:
            logger.error(f"Error processing sample at index {idx}: {e}", exc_info=True)
            continue


def compute_metrics_from_data(
    targets: TensorDict,
    predictions: TensorDict,
    metric_names: Optional[List[str]] = ["L2", "LSIM"],
) -> Tuple[Dict, Dict]:
    """
    Computes metrics from already-loaded targets and predictions TensorDicts.

    Returns:
        A tuple of (results_summary, raw_errors_bt):
        - results_summary: Dict of {metric: {var: (mean, std)}}
        - raw_errors_bt: Dict of {metric: {var: [B, T] tensor}}
    """
    if targets.is_empty():
        logger.warning("compute_metrics_from_data received empty targets. Skipping.")
        return {}, {}

    # Define the variables of interest for metric computation
    loss_relevant_fields = ["v_x", "v_y", "p"]

    # Filter targets and predictions to only contain relevant fields
    # This ensures 'variable_mode=all' only averages over these fields
    try:
        # Check which keys are actually available in *both* tensordicts
        available_keys_in_targets = set(targets.keys())
        available_keys_in_preds = set(predictions.keys())

        available_fields = [
            k
            for k in loss_relevant_fields
            if k in available_keys_in_targets and k in available_keys_in_preds
        ]

        missing_keys = [k for k in loss_relevant_fields if k not in available_fields]

        if missing_keys:
            logger.warning(
                f"Missing fields in dataset: {missing_keys}. Proceeding with available fields."
            )

        if not available_fields:
            logger.error("No loss_relevant_fields found in data. Aborting evaluation.")
            return {}, {}

        # Now select only the keys that are guaranteed to exist
        targets_filtered = targets.select(*available_fields)
        predictions_filtered = predictions.select(*available_fields)

        # Update loss_relevant_fields to only include what's available for the rest of the function
        loss_relevant_fields = available_fields

    except KeyError as e:
        logger.error(f"Missing one of the required fields. Error: {e}", exc_info=True)
        return {}, {}

    results_summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    raw_errors_bt: Dict[str, Dict[str, torch.Tensor]] = {}

    assert metric_names is not None
    for metric_name in metric_names:
        logger.info(f"Computing metric: {metric_name}")
        results_summary[metric_name] = {}
        raw_errors_bt[metric_name] = {}

        # 1. Compute per-variable metrics
        for var in loss_relevant_fields:
            try:
                metric_fn_var = Metric(
                    mode=metric_name, variable_mode="single", variable_name=var
                )
                # metric_fn returns [B, T] tensor
                distances_bt = metric_fn_var(targets_filtered, predictions_filtered)

                if distances_bt.numel() == 0:
                    logger.warning(
                        f"Metric {metric_name} for {var} returned no results."
                    )
                    continue

                # Save the raw [B, T] tensor of errors
                raw_errors_bt[metric_name][var] = distances_bt

                # Average over time to get one value per batch sample [B]
                distances_b = distances_bt.mean(dim=1)

                # Store mean and std over the batch for the summary
                results_summary[metric_name][var] = (
                    distances_b.mean().item(),
                    distances_b.std().item(),
                )
            except Exception as e:
                logger.error(
                    f"Failed to compute metric {metric_name} for {var}: {e}",
                    exc_info=True,
                )

        # 2. Compute "all" (averaged over variables) metric
        if len(loss_relevant_fields) > 1:
            try:
                metric_fn_all = Metric(mode=metric_name, variable_mode="all")

                # metric_fn_all returns [B, T] tensor (averaged over variables)
                distances_all_bt = metric_fn_all(targets_filtered, predictions_filtered)

                if distances_all_bt.numel() == 0:
                    logger.warning(
                        f"Metric {metric_name} for 'all' returned no results."
                    )
                    continue

                # Save the raw [B, T] tensor of errors
                raw_errors_bt[metric_name]["all"] = distances_all_bt

                # Average over time to get one value per batch sample [B]
                distances_all_b = distances_all_bt.mean(dim=1)

                # Store mean and std over the batch for the summary
                results_summary[metric_name]["all"] = (
                    distances_all_b.mean().item(),
                    distances_all_b.std().item(),
                )
            except Exception as e:
                logger.error(
                    f"Failed to compute metric {metric_name} for 'all': {e}",
                    exc_info=True,
                )
        elif len(loss_relevant_fields) == 1:
            # If only one field, "all" is the same as the single variable
            var_name = loss_relevant_fields[0]
            if var_name in results_summary[metric_name]:
                results_summary[metric_name]["all"] = results_summary[metric_name][
                    var_name
                ]
                raw_errors_bt[metric_name]["all"] = raw_errors_bt[metric_name][var_name]

    logger.info(f"Metric computation complete. Results: {results_summary}")
    return results_summary, raw_errors_bt


def run_evaluation(
    model: torch.nn.Module,
    loader,
    output_len: int,
    rollout_fn: RolloutFn,
    metric_names: Optional[List[str]] = ["L2", "LSIM"],
    save_individual_dir: Optional[str] = None,  # New argument
) -> Tuple[Dict, Dict, TensorDict, TensorDict]:
    """
    Orchestrates the entire evaluation pipeline for a given model
    and computes metrics per-variable and averaged across variables.

    Optionally saves individual rollouts to `save_individual_dir`.

    Returns:
        A tuple of (metrics_summary, raw_errors_bt, targets, predictions)
    """
    rollout_name = rollout_fn.__name__ if rollout_fn is not None else "<undefined>"
    logger.info(
        f"Starting evaluation for model '{model.__class__.__name__}' using '{rollout_name}'..."
    )

    all_targets, all_predictions = [], []
    prediction_generator = generate_predictions_for_dataset(
        model=model,
        dataset=loader.dataset,
        output_len=output_len,
        rollout_fn=rollout_fn,
    )

    logger.info("Generating predictions...")
    if save_individual_dir:
        logger.info(f"Saving individual rollouts to: {save_individual_dir}")

    for gt, pred, metadata, idx in prediction_generator:
        # --- Save individual rollout logic ---
        if save_individual_dir:
            try:
                # Extract Reynolds number for filename
                cond_val_raw = metadata.get("cond_target", [0.0])[0]
                if torch.is_tensor(cond_val_raw) and cond_val_raw.numel() > 0:
                    cond_val = cond_val_raw.item()
                elif isinstance(cond_val_raw, (int, float)):
                    cond_val = cond_val_raw
                else:
                    cond_val = 0.0  # Default

                # Format for filename, e.g., "Re550p00_sample0001.pt"
                re_str = f"Re{cond_val:.2f}".replace(".", "p")
                filename = f"{re_str}_sample{idx:04d}.pt"

                save_path = os.path.join(save_individual_dir, filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                torch.save(
                    {
                        "target": gt.cpu(),
                        "prediction": pred.cpu(),
                        "metadata": metadata,
                    },
                    save_path,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to save individual rollout for sample {idx}: {e}"
                )
        # --- End of save logic ---

        # Keep collecting for aggregate metrics
        all_targets.append(gt)
        all_predictions.append(pred)

    if not all_predictions:
        logger.warning(
            "generate_predictions_for_dataset returned no data. Skipping evaluation."
        )
        empty_td = TensorDict({}, batch_size=[0])
        return {}, {}, empty_td, empty_td

    targets = stack_tensordict(all_targets)
    predictions = stack_tensordict(all_predictions)

    logger.info(f"Generated predictions for {targets.shape[0]} samples.")

    metrics_summary, raw_errors_bt = compute_metrics_from_data(
        targets, predictions, metric_names
    )

    logger.info("Evaluation complete.")
    return metrics_summary, raw_errors_bt, targets, predictions


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

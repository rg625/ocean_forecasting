# models/metrics_utils.py

from tensordict import TensorDict, stack as stack_tensordict
import torch
from typing import Dict, Tuple, List, Optional
import logging
from models.metrics import Metric

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_all_metrics(
    target: TensorDict,
    prediction: TensorDict,
    loader,
    variables: List[str],
    custom_min_max: Optional[Dict[str, Tuple[float, float]]] = None,
    chunk_size: int = 8,  # limit batch size in normalization
) -> dict:
    """
    Computes all metrics (L2, SSIM, PSNR, VI) for each variable and for 'all'.
    Applies manual normalization for custom-computed variables like 'vort'.

    Args:
        target (TensorDict): Ground truth data (denormalized).
        prediction (TensorDict): Model predictions (denormalized).
        loader: The dataset or loader that contains to_unit_range().
        variables (List[str]): Variables to evaluate.
        custom_min_max (dict, optional): For manual normalization, e.g. {'vort': (min, max)}

    Returns:
        dict: Nested structure {metric_mode: {variable_name: (mean, std)}}
    """
    results: Dict = {}

    # --- Prepare data ---
    target_norm = {}
    prediction_norm = {}

    for var in variables:
        if custom_min_max and var in custom_min_max:
            vmin, vmax = custom_min_max[var]
            target_norm[var] = (target[var] - vmin) / (vmax - vmin + 1e-8)
            prediction_norm[var] = (prediction[var] - vmin) / (vmax - vmin + 1e-8)
        else:
            # Process in smaller CPU chunks to avoid GPU OOM
            target_chunks = []
            pred_chunks = []
            for start in range(0, target.shape[0], chunk_size):
                end = start + chunk_size
                t_chunk = loader.to_unit_range(target[start:end].cpu())[
                    var
                ]  # keep on CPU
                p_chunk = loader.to_unit_range(prediction[start:end].cpu())[var]
                target_chunks.append(t_chunk)
                pred_chunks.append(p_chunk)
            target_norm[var] = torch.cat(target_chunks, dim=0)  # .to(DEVICE)
            prediction_norm[var] = torch.cat(pred_chunks, dim=0)  # .to(DEVICE)

    target_td = TensorDict(target_norm, batch_size=target.batch_size)
    pred_td = TensorDict(prediction_norm, batch_size=prediction.batch_size)

    for mode in Metric.VALID_MODES:
        results[mode] = {}
        for var in variables + ["all"]:
            variable_mode = "all" if var == "all" else "single"
            if var == "v_x" and variable_mode == "single" and mode in ["L2", "LSIM"]:
                metric_fn = Metric(
                    mode=mode,
                    variable_mode=variable_mode,
                    variable_name=None if variable_mode == "all" else var,
                )
                dist = metric_fn(target_td, pred_td)  # [B, T]
                results[mode][var] = (dist.mean().item(), dist.std().item())
            else:
                pass
    return results


def compute_vorticity(vx, vy, chunk_size=10):
    vort_list = []
    for i in range(0, vx.size(0), chunk_size):
        vx_chunk = vx[i : i + chunk_size]
        vy_chunk = vy[i : i + chunk_size]
        vxDx, vxDy = torch.gradient(vx_chunk, dim=(1, 2))
        vyDx, vyDy = torch.gradient(vy_chunk, dim=(1, 2))
        vort_list.append(vxDx - vyDy)
    return torch.cat(vort_list, dim=0)


def compute_vorticity_diff(vx, vy, chunk_size=10):
    vort_list = []
    for i in range(0, vx.size(0), chunk_size):
        vx_chunk = vx[i : i + chunk_size]
        vy_chunk = vy[i : i + chunk_size]
        vxDx, vxDy = torch.gradient(vx_chunk, dim=(1, 2))
        vyDx, vyDy = torch.gradient(vy_chunk, dim=(1, 2))
        vort_list.append(vxDy - vyDx)
    return torch.cat(vort_list, dim=0)


def run_long_rollout(model, input_seq, rollout_steps):
    input_seq = input_seq.unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]

    with torch.no_grad():
        out = model(input_seq, seq_length=rollout_steps)

    return out  # [T+rollout_steps, C, H, W]


def exhaustive_predictions(model, dataset, input_len, output_len):
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Make sure your logger prints DEBUG messages

    all_final_targets = []
    all_final_predictions = []

    logger.debug(
        f"Starting exhaustive_predictions with input_len={input_len}, output_len={output_len}"
    )
    logger.debug(f"Dataset length: {len(dataset)}")

    for idx in range(0, len(dataset), (output_len + input_len)):
        logger.debug(f"\n--- Processing sample index: {idx} ---")

        try:
            # Run rollout
            input_seq, ground_truth_future, _metadata = dataset[idx]
            input_seq["obstacle_mask"] = _metadata["obstacle_mask"][0].repeat(
                *input_seq.batch_size, 1, 1
            )
            input_seq["Re_input"] = _metadata["Re_input"][0].repeat(
                *input_seq.batch_size
            )

            logger.debug(f"input_seq shape: {input_seq.shape}")
            logger.debug(f"target_seq shape: {ground_truth_future.shape}")

            # Run rollout with input_seq and rollout_steps = output_len
            predicted_future_seq = run_long_rollout(
                model=model, input_seq=input_seq, rollout_steps=output_len
            ).x_preds
            logger.debug("Rollout done.")

            # Log raw tensordict info
            logger.debug(f"Predicted future sequence: {predicted_future_seq}")
            logger.debug(f"Ground truth future sequence: {ground_truth_future}")

            # Denormalize predictions and ground truth
            logger.debug("Denormalizing predicted_future_seq")
            predicted_future_seq = dataset.denormalize(predicted_future_seq)
            logger.debug(f"Denormalized predicted_future_seq: {predicted_future_seq}")

            logger.debug("Denormalizing ground_truth_future")
            ground_truth_future = dataset.denormalize(ground_truth_future.unsqueeze(0))
            logger.debug(f"Denormalized ground_truth_future: {ground_truth_future}")

            # Log batch sizes and shapes
            pred_batch_size = (
                predicted_future_seq.batch_size
                if hasattr(predicted_future_seq, "batch_size")
                else "N/A"
            )
            targ_batch_size = (
                ground_truth_future.batch_size
                if hasattr(ground_truth_future, "batch_size")
                else "N/A"
            )
            logger.debug(f"predicted_future_seq batch_size: {pred_batch_size}")
            logger.debug(f"ground_truth_future batch_size: {targ_batch_size}")

            # Check length consistency
            pred_len = (
                pred_batch_size[1]
                if pred_batch_size and len(pred_batch_size) > 1
                else None
            )
            targ_len = (
                targ_batch_size[1]
                if targ_batch_size and len(targ_batch_size) > 1
                else None
            )
            logger.debug(f"pred_len: {pred_len}, targ_len: {targ_len}")

            if pred_len != output_len or targ_len != output_len:
                logger.warning(
                    f"Length mismatch at sample {idx}: pred_len={pred_len}, targ_len={targ_len}"
                )

            all_final_predictions.append(predicted_future_seq)
            all_final_targets.append(ground_truth_future)

        except Exception as e:
            logger.error(f"Exception at sample index {idx}: {e}", exc_info=True)
            continue

    if not all_final_predictions or not all_final_targets:
        logger.error(
            "No valid predictions or targets collected. Returning empty TensorDicts."
        )
        return TensorDict({}, batch_size=[0]), TensorDict({}, batch_size=[0])

    logger.debug("Stacking all predictions and targets now...")

    try:
        stacked_targets = stack_tensordict(all_final_targets, dim=0).squeeze()
        stacked_predictions = stack_tensordict(all_final_predictions, dim=0).squeeze()
    except Exception as e:
        logger.error(f"Error while stacking tensordicts: {e}", exc_info=True)
        raise

    logger.debug(
        f"Stacked targets batch size: {stacked_targets.batch_size if hasattr(stacked_targets, 'batch_size') else 'N/A'}"
    )
    logger.debug(
        f"Stacked predictions batch size: {stacked_predictions.batch_size if hasattr(stacked_predictions, 'batch_size') else 'N/A'}"
    )

    logger.info("exhaustive_predictions completed successfully.")

    return stacked_targets, stacked_predictions


def run_full_evaluation_and_report(model, loader, input_len: int, output_len: int):
    """
    Orchestrates the entire evaluation pipeline using configurable
    input sequence length and output sequence length.
    """
    # --- 1. Generate predictions and ground truths ---
    batched_targets, batched_predictions = exhaustive_predictions(
        model=model,
        dataset=loader.dataset,
        input_len=input_len,
        output_len=output_len,
    )

    if batched_targets.is_empty():
        logger.error("Prediction generation failed. Cannot proceed.")
        return

    # --- 2. Post-process variables ---
    logger.info("Post-processing to compute derived variables...")
    custom_min_max = {}
    if "v_x" in batched_targets and "v_y" in batched_targets:
        vort_truth = compute_vorticity_diff(
            batched_targets["v_x"], batched_targets["v_y"]
        )
        batched_targets["vort"] = vort_truth

        vort_pred = compute_vorticity_diff(
            batched_predictions["v_x"].squeeze(), batched_predictions["v_y"].squeeze()
        )
        batched_predictions["vort"] = vort_pred

        global_vort_min = vort_truth.min().item()
        global_vort_max = vort_truth.max().item()
        custom_min_max["vort"] = (global_vort_min, global_vort_max)

        logger.info(
            f"Global vorticity range: [{global_vort_min:.4f}, {global_vort_max:.4f}]"
        )
    else:
        logger.warning("Variables 'v_x' and 'v_y' not found. Skipping vorticity calc.")

    # --- 3. Compute metrics ---
    vars_to_eval = [
        k
        for k in batched_predictions.keys()
        if k in batched_targets
        and k not in ["seq_length", "Re_target", "Re_input", "obstacle_mask"]
    ]
    logger.info(f"Computing metrics for: {vars_to_eval}")

    metrics_result = compute_all_metrics(
        target=batched_targets,
        prediction=batched_predictions,
        loader=loader,
        variables=vars_to_eval,
        custom_min_max=custom_min_max,
    )
    return metrics_result


def metrics_to_latex_table(metrics: dict) -> str:
    """
    Convert metrics dict to a LaTeX table showing mean ± std for each variable and mode.
    Small values are shown in scientific notation for better readability.
    """
    all_vars = sorted({v for mode_vals in metrics.values() for v in mode_vals.keys()})
    all_modes = Metric.VALID_MODES

    header = r"\begin{tabular}{l" + "c" * len(all_modes) + "}\n"
    header += "Variable & " + " & ".join(all_modes) + r" \\\hline" + "\n"

    def fmt(num):
        if num != num:  # NaN check
            return "nan"
        return f"{num:.2e}"

    rows = []
    for var in all_vars:
        row = [var]
        for mode in all_modes:
            mean, std = metrics.get(mode, {}).get(var, (float("nan"), float("nan")))
            row.append(f"{fmt(mean)} ± {fmt(std)}")
        rows.append(" & ".join(row) + r" \\")

    table = header + "\n".join(rows) + "\n\\end{tabular}"
    return table


def convert_to_tensordict_fields(tensor, var_names):
    """
    Convert tensor (num_samples, seq_len, channels, H, W)
    to TensorDict with fields {var_name: (time, H, W)} where time = num_samples * seq_len
    """
    num_samples, seq_len, channels, H, W = tensor.shape
    # reshape so time dimension is samples * seq_len
    tensor_reshaped = tensor.permute(
        0, 1, 3, 4, 2
    )  # .reshape(num_samples * seq_len, H, W, channels)
    td_fields = {}
    for i, var in enumerate(var_names):
        td_fields[var] = tensor_reshaped[..., i]
    return TensorDict(td_fields, batch_size=[num_samples, seq_len])


def exhaustive_diffusion_rollout(
    model, loader, device, numSamples=5, sequenceLength=60
):
    """
    Run exhaustive rollout on all dataset samples using the diffusion model.
    Returns TensorDicts with separate fields per variable.
    """
    model.eval()
    all_gt = []
    all_pred = []

    with torch.no_grad():
        for idx, sample in enumerate(loader):
            # Original batch is 1; repeat for numSamples
            d = (
                sample["data"].to(device).repeat(numSamples, 1, 1, 1, 1)
            )  # (numSamples, seq_len, C, H, W)
            # print(f"testLoader.shape: {len(testLoader)}")
            batch_size, seq_len, C, H, W = d.shape

            prediction = torch.zeros(
                [batch_size, sequenceLength, C, H, W], device=device
            )
            inputSteps = 2

            # Copy first inputSteps from data (no prediction)
            for i in range(inputSteps):
                prediction[:, i] = d[:, i]

            # Auto-regressive rollout
            for i in range(inputSteps, sequenceLength):
                cond = []
                for j in range(inputSteps, 0, -1):
                    cond += [prediction[:, i - j : i - (j - 1)]]  # collect input steps
                cond = torch.concat(cond, dim=2)  # combine along channel dimension

                result = model(
                    conditioning=cond, data=d[:, i - 1 : i]
                )  # auto-regressive inference
                result[:, :, -1:] = d[
                    :, i : i + 1, -1:
                ]  # replace simparam prediction with true values
                prediction[:, i : i + 1] = result

            # Move tensors to CPU for stacking and conversion
            prediction_cpu = prediction.cpu()
            gt_cpu = d.cpu()

            all_pred.append(prediction_cpu)
            all_gt.append(gt_cpu)

    # Concatenate all samples across dataset dimension (axis=1 because batch size is at dim=1 in your example)
    all_pred = torch.cat(all_pred, dim=0)  # shape: (numSamples, total_seq_len, C, H, W)
    all_gt = torch.cat(all_gt, dim=0)
    var_names = [
        "v_x",
        "v_y",
        "p",
    ]  # adjust if you have more channels or different order

    # Convert to TensorDict format with shape (time, H, W)
    pred_td = convert_to_tensordict_fields(all_pred.transpose(-2, -1), var_names)
    gt_td = convert_to_tensordict_fields(all_gt.transpose(-2, -1), var_names)

    return gt_td, pred_td


def run_full_eval_and_report_diffusion(
    model, loader, device, numSamples=5, sequenceLength=60
):
    """
    Orchestrates evaluation pipeline for diffusion model using exhaustive rollout.

    Args:
        model: Diffusion model.
        loader: DataLoader with dataset for evaluation.
        device: torch device (CPU or CUDA).
        numSamples: Number of diffusion samples per sequence.
        sequenceLength: Total sequence length to rollout.

    Returns:
        metrics_result: dict of metrics like L2, SSIM, PSNR, VI.
    """
    # --- 1. Generate predictions and ground truths ---
    logger.info(
        f"Running exhaustive diffusion rollout: numSamples={numSamples}, sequenceLength={sequenceLength}"
    )
    batched_targets, batched_predictions = exhaustive_diffusion_rollout(
        model=model,
        loader=loader[0],
        device=device,
        numSamples=numSamples,
        sequenceLength=sequenceLength,
    )

    if batched_targets.is_empty():
        logger.error("Diffusion prediction generation failed. Cannot proceed.")
        return

    # --- 2. Post-process variables ---
    logger.info("Post-processing to compute derived variables for diffusion model...")
    custom_min_max = {}
    if "v_x" in batched_targets and "v_y" in batched_targets:
        vort_truth = compute_vorticity_diff(
            batched_targets["v_x"], batched_targets["v_y"]
        )
        batched_targets["vort"] = vort_truth

        vort_pred = compute_vorticity_diff(
            batched_predictions["v_x"].squeeze(), batched_predictions["v_y"].squeeze()
        )
        batched_predictions["vort"] = vort_pred

        global_vort_min = vort_truth.min().item()
        global_vort_max = vort_truth.max().item()
        custom_min_max["vort"] = (global_vort_min, global_vort_max)

        logger.info(
            f"Global vorticity range (diffusion): [{global_vort_min:.4f}, {global_vort_max:.4f}]"
        )
    else:
        logger.warning("Variables 'v_x' and 'v_y' not found. Skipping vorticity calc.")

    # --- 3. Compute metrics ---
    vars_to_eval = [
        k
        for k in batched_predictions.keys()
        if k in batched_targets
        and k not in ["seq_length", "Re_target", "Re_input", "obstacle_mask"]
    ]
    logger.info(f"Computing metrics for diffusion model: {vars_to_eval}")

    metrics_result = compute_all_metrics(
        target=batched_targets,
        prediction=batched_predictions,
        loader=loader[1],
        variables=vars_to_eval,
        custom_min_max=custom_min_max,
    )
    return metrics_result

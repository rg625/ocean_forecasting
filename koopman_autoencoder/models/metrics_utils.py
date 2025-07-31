# models/metrics_utils.py

from tensordict import TensorDict, stack as stack_tensordict
import torch
from typing import Dict, Tuple, List, Optional
import logging
from models.metrics import Metric
from tqdm import tqdm

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
            # Use manual normalization
            vmin, vmax = custom_min_max[var]
            target_norm[var] = (target[var] - vmin) / (vmax - vmin + 1e-8)
            prediction_norm[var] = (prediction[var] - vmin) / (vmax - vmin + 1e-8)
        else:
            # Use loader's to_unit_range
            target_norm[var] = loader.to_unit_range(target)[var]
            prediction_norm[var] = loader.to_unit_range(prediction)[var]

    # Convert to TensorDict
    target_td = TensorDict(target_norm, batch_size=target.batch_size)
    pred_td = TensorDict(prediction_norm, batch_size=prediction.batch_size)

    # --- Run metrics ---
    for mode in Metric.VALID_MODES:
        results[mode] = {}

        for var in variables + ["all"]:
            variable_mode = "all" if var == "all" else "single"
            metric_fn = Metric(
                mode=mode,
                variable_mode=variable_mode,
                variable_name=None if variable_mode == "all" else var,
            )

            dist = metric_fn(target_td, pred_td)  # [B, T]
            results[mode][var] = (dist.mean().item(), dist.std().item())

    return results


def compute_vorticity(vx, vy):
    vxDx, vxDy = torch.gradient(vx, dim=(1, 2))  # vx: [T, H, W]
    vyDx, vyDy = torch.gradient(vy, dim=(1, 2))
    # return vyDx - vxDy  # [T, H, W]
    return vxDx - vyDy  # [T, H, W]


def run_long_rollout(model, input_seq, rollout_steps):
    input_seq = input_seq.unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]

    with torch.no_grad():
        out = model(input_seq, seq_length=rollout_steps)

    return out  # [T+rollout_steps, C, H, W]


def exhaustive_predictions(
    model, loader, prediction_steps
) -> Tuple[TensorDict, TensorDict]:
    """
    Generates exhaustive predictions for a given number of steps and returns
    ONLY THE FINAL FRAME of each prediction and its corresponding ground truth.
    """
    logger.info(
        f"Starting exhaustive last-frame prediction for step {prediction_steps}..."
    )
    model.eval()

    dataset = loader.dataset
    if not hasattr(dataset, "raw_data_td"):
        raise TypeError(
            "The loader's dataset must be a custom QGDataset for this evaluation."
        )

    all_final_predictions, all_final_targets = [], []

    # We iterate through the dataset using its length
    pbar = tqdm(
        range(len(dataset)),
        desc=f"Generating Last-Frame Predictions (T={prediction_steps})",
    )
    with torch.no_grad():
        for i in pbar:
            # Get the input sequence and the full ground truth sequence
            input_seq, ground_truth_full_seq, metadata = dataset[i, prediction_steps]

            input_seq["obstacle_mask"] = metadata["obstacle_mask"][0].repeat(
                *input_seq.batch_size, 1, 1
            )
            if "Re_input" in metadata.keys():
                input_seq["Re_input"] = (
                    metadata["Re_input"][0].view(-1).repeat(*input_seq.batch_size)
                )

            # Use the dedicated rollout function to get model predictions
            rollout_result = run_long_rollout(model, input_seq, prediction_steps)
            predicted_full_seq = rollout_result.x_preds.cpu()

            # --- Extract only the LAST frame ---
            final_predicted_frame = predicted_full_seq[:, -1]
            final_target_frame = ground_truth_full_seq[-1]

            # Denormalize and collect the single final frames
            all_final_predictions.append(dataset.denormalize(final_predicted_frame))
            all_final_targets.append(
                dataset.denormalize(final_target_frame.unsqueeze(0))
            )  # Add batch dim

    pbar.close()

    if not all_final_predictions:
        logger.warning("No predictions were generated.")
        return TensorDict({}, batch_size=[0]), TensorDict({}, batch_size=[0])

    logger.info("Stacking all final-frame predictions and targets...")
    # The result will have a time dimension of 1
    return (
        stack_tensordict(all_final_targets, dim=0).squeeze(),
        stack_tensordict(all_final_predictions, dim=0).squeeze(),
    )


def run_full_evaluation_and_report(model, loader, prediction_steps):
    """
    Orchestrates the entire evaluation pipeline using the last-frame prediction strategy.
    """
    # --- 1. Generate all predictions and ground truths ---
    batched_targets, batched_predictions = exhaustive_predictions(
        model=model,
        loader=loader,
        prediction_steps=prediction_steps,
    )

    if batched_targets.is_empty():
        logger.error("Prediction generation failed. Cannot proceed.")
        return

    # --- 2. Post-process the full data batch ---
    logger.info("Post-processing full data batch to compute on-the-fly variables...")
    custom_min_max = {}

    if "v_x" in batched_targets and "v_y" in batched_targets:
        vort_truth = compute_vorticity(batched_targets["v_x"], batched_targets["v_y"])
        batched_targets["vort"] = vort_truth

        vort_pred = compute_vorticity(
            batched_predictions["v_x"].squeeze(), batched_predictions["v_y"].squeeze()
        )
        batched_predictions["vort"] = vort_pred

        global_vort_min = vort_truth.min().item()
        global_vort_max = vort_truth.max().item()
        custom_min_max["vort"] = (global_vort_min, global_vort_max)

        logger.info(
            f"Global vorticity range for scaling: [{global_vort_min:.4f}, {global_vort_max:.4f}]"
        )
    else:
        logger.warning(
            "Variables 'v_x' and 'v_y' not found. Skipping vorticity calculation."
        )

    # --- 3. Compute final metrics ---
    vars_to_eval = [
        k
        for k in batched_predictions.keys()
        if k in batched_targets
        and k not in ["seq_length", "Re_target", "Re_input", "obstacle_mask"]
    ]
    logger.info(f"Computing metrics for variables: {vars_to_eval}")

    metrics_result = compute_all_metrics(
        target=batched_targets.unsqueeze(1),
        prediction=batched_predictions.unsqueeze(1),
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

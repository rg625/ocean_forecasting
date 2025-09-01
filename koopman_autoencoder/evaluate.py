import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
import logging
from omegaconf import OmegaConf
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
from tensordict import TensorDict
import seaborn as sns
from typing import Dict, Tuple, List, Optional


# Import necessary functions and classes from your project
# KAE specific imports
from models.config_classes import Config
from models.autoencoder import KoopmanAutoencoder
from models.dataloader import create_dataloaders
from models.utils import load_checkpoint, load_datasets
from models.metrics import Metric

# Import the new, refactored metrics utils
from models.metrics_utils import (
    run_full_evaluation_and_report,
    compute_vorticity,
    run_full_eval_and_report_diffusion,
)

# Diffusion Model specific imports
from turbpred.params import DataParams, ModelParamsDecoder
from turbpred.model_diffusion import DiffusionModel
from turbpred.turbulence_dataset import TurbulenceDataset
from turbpred.data_transformations import Transforms as DataTransforms

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid")
cmap = sns.color_palette("icefire", as_cmap=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# #############################################################################
# LOCAL, FIXED VERSIONS OF METRICS UTILS
# To resolve the runtime error, the following functions from metrics_utils.py
# have been integrated directly into this script with the necessary fix.
# #############################################################################


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
    """
    results: Dict = {}
    target_norm = {}
    prediction_norm = {}

    for var in variables:
        if custom_min_max and var in custom_min_max:
            vmin, vmax = custom_min_max[var]
            target_norm[var] = (target[var] - vmin) / (vmax - vmin + 1e-8)
            prediction_norm[var] = (prediction[var] - vmin) / (vmax - vmin + 1e-8)
        else:
            target_chunks = []
            pred_chunks = []
            # Ensure we are using the correct loader (val_loader for normalization stats)
            norm_loader = loader if not isinstance(loader, list) else loader[1]
            for start in range(0, target.shape[0], chunk_size):
                end = start + chunk_size
                t_chunk = norm_loader.dataset.to_unit_range(target[start:end].cpu())[
                    var
                ]
                p_chunk = norm_loader.dataset.to_unit_range(
                    prediction[start:end].cpu()
                )[var]
                target_chunks.append(t_chunk)
                pred_chunks.append(p_chunk)
            target_norm[var] = torch.cat(target_chunks, dim=0)
            prediction_norm[var] = torch.cat(pred_chunks, dim=0)

    target_td = TensorDict(target_norm, batch_size=target.batch_size).to(
        prediction.device
    )
    pred_td = TensorDict(prediction_norm, batch_size=prediction.batch_size).to(
        prediction.device
    )

    for mode in Metric.VALID_MODES:
        results[mode] = {}
        for var in variables + ["all"]:
            variable_mode = "all" if var == "all" else "single"
            metric_fn = Metric(
                mode=mode,
                variable_mode=variable_mode,
                variable_name=None if variable_mode == "all" else var,
            )
            dist = metric_fn(target_td, pred_td)
            results[mode][var] = (dist.mean().item(), dist.std().item())
    return results


def convert_to_tensordict_fields(tensor, var_names):
    """
    Convert tensor (num_samples, seq_len, channels, H, W)
    to TensorDict with fields {var_name: (time, H, W)}
    """
    num_samples, seq_len, channels, H, W = tensor.shape
    tensor_reshaped = tensor.permute(0, 1, 3, 4, 2)
    td_fields = {}
    for i, var in enumerate(var_names):
        td_fields[var] = tensor_reshaped[..., i]
    return TensorDict(td_fields, batch_size=[num_samples, seq_len])


def load_kae_model(model_config, device):
    """Loads only the KAE model based on its config."""
    config_path = model_config["config_path"]
    checkpoint_path = model_config["checkpoint_path"]

    base_config = OmegaConf.structured(Config)
    file_config = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base_config, file_config)
    OmegaConf.resolve(cfg)

    model = KoopmanAutoencoder(
        data_variables=cfg.data.variables,
        input_frames=cfg.data.input_sequence_length,
        height=cfg.model.height,
        width=cfg.model.width,
        latent_dim=cfg.model.latent_dim,
        re_embedding_dim=cfg.model.re_embedding_dim,
        re_cond_type=cfg.model.re_cond_type,
        operator_mode=cfg.model.operator_mode,
        hidden_dims=cfg.model.hidden_dims,
        transformer_config=cfg.model.transformer,
        use_checkpoint=False,
        predict_re=cfg.model.predict_re,
        re_grad_enabled=cfg.model.re_grad_enabled,
        residual=cfg.model.residual,
        **cfg.model.conv_kwargs,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr_scheduler.lr)
    model, _, _, _ = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer)
    model.eval()
    return model, cfg


def load_acdm_model_and_loader(model_config, device):
    """Loads the ACDM model and its specific test loader for rollouts."""
    checkpoint_path = model_config["checkpoint_path"]
    model_type = model_config["model_type"]
    simFields = [
        "pres"
    ]  # the provided data set contains velocity (implict), as well as density and pressure values
    simParams = ["rey"]
    sequenceLength = [60, 2]

    p_md = ModelParamsDecoder(
        arch="direct-ddpm+Prev",
        diffSteps=20,
        diffSchedule="linear",
        diffCondIntegration="noisy",
        trainingNoise=0.0,
    )
    p_d = DataParams(
        batch=1,
        augmentations=["normalize"],
        sequenceLength=[2, 60],
        randSeqOffset=False,
        dataSize=[128, 64],
        dimension=2,
        simFields=["pres"],
        simParams=["rey"],
        normalizeMode="incMixed",
    )

    testSet = TurbulenceDataset(
        "Test",
        dataDirs=["/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/data"],
        filterTop=["128_inc"],
        filterSim=[[0]],
        excludefilterSim=False,
        filterFrame=[(20, 1300)],
        sequenceLength=[sequenceLength],
        randSeqOffset=False,
        simFields=simFields,
        simParams=simParams,
        printLevel="sim",
    )
    testSet.transform = DataTransforms(p_d)
    test_loader = DataLoader(
        testSet, batch_size=1, drop_last=False, sampler=SequentialSampler(testSet)
    )

    condChannels = 2 * (2 + len(p_d.simFields) + len(p_d.simParams))
    model = DiffusionModel(p_d, p_md, dimension=0, condChannels=condChannels)
    model.training = False
    model.inferenceConditioningIntegration = (
        "noisy" if model_type.lower() == "acdm" else "clean"
    )

    loaded = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(loaded["stateDictDecoder"], strict=True)
    model.to(device)

    return model, test_loader


def run_rollout_for_plots_kae(model, loader, rollout_steps, device):
    """Generates a single rollout prediction for plotting KAE models."""
    input_seq, gt_future, metadata = loader.dataset[500, rollout_steps]
    input_seq["obstacle_mask"] = metadata["obstacle_mask"][0].repeat(
        *input_seq.batch_size, 1, 1
    )
    input_seq["Re_input"] = metadata["Re_input"][0].repeat(*input_seq.batch_size)

    with torch.no_grad():
        predicted_td = model.rollout(input_seq.to(device), steps=rollout_steps)

    gt_denorm = loader.dataset.denormalize(gt_future)
    pred_denorm = loader.dataset.denormalize(predicted_td.cpu())

    return gt_denorm, pred_denorm, metadata


def run_rollout_for_plots_diffusion(model, loader, rollout_steps, device):
    """Generates a single rollout prediction for plotting diffusion models."""
    sample = next(iter(loader))
    data = sample["data"].to(device)

    batch_size, _, C, H, W = data.shape
    prediction = torch.zeros([batch_size, rollout_steps, C, H, W], device=device)
    input_steps = 2
    gt_sim_params = data[:, 0:1, -1:]

    prediction[:, :input_steps] = data[:, :input_steps]

    with torch.no_grad():
        for i in range(input_steps, rollout_steps):
            cond = torch.cat(
                [prediction[:, i - j : i - (j - 1)] for j in range(input_steps, 0, -1)],
                dim=2,
            )
            result = model(conditioning=cond, data=data[:, i - 1 : i])
            result[:, :, -1:] = gt_sim_params
            prediction[:, i : i + 1] = result

    var_names = ["v_x", "v_y", "p"]
    gt_td = TensorDict(
        {
            k: v.squeeze(1)
            for k, v in zip(var_names, data.squeeze(0).cpu().split(1, dim=2))
        },
        batch_size=data.shape[1],
    )
    pred_td = TensorDict(
        {
            k: v.squeeze(1)
            for k, v in zip(var_names, prediction.squeeze(0).cpu().split(1, dim=2))
        },
        batch_size=prediction.shape[1],
    )

    return gt_td, pred_td, None


def generate_joint_rollout_plots(all_rollouts, variables, frame_stride=5):
    """
    Generates and saves joint rollout plots for all models and variables.
    """
    if not all_rollouts:
        logger.warning("No rollout data to plot.")
        return

    model_names = list(all_rollouts.keys())
    gt_dict = all_rollouts[model_names[0]]["gt"]
    re_number = all_rollouts[model_names[0]]["metadata"].get("Re_input", [1000])[0]

    if "v_x" in gt_dict.keys() and "v_y" in gt_dict.keys():
        gt_dict["vort"] = compute_vorticity(gt_dict["v_x"], gt_dict["v_y"])

    for var in variables:
        if var not in gt_dict.keys():
            logger.warning(f"Variable '{var}' not in ground truth. Skipping plot.")
            continue

        logger.info(f"Generating plot for variable: {var}")

        gt_seq = gt_dict[var]
        num_frames = gt_seq.shape[0]
        indices = list(range(0, num_frames, frame_stride))
        num_plots = min(len(indices), 10)

        num_rows = 1 + 2 * len(model_names)
        fig = plt.figure(figsize=(2.5 * num_plots, 2.5 * num_rows))
        spec = gridspec.GridSpec(num_rows, num_plots)

        for i, idx in enumerate(indices[:num_plots]):
            ax = fig.add_subplot(spec[0, i])
            _ = ax.imshow(gt_seq[idx].cpu().T, cmap=cmap, origin="lower")
            ax.axis("off")
            ax.set_title(f"t={idx}", fontsize=10)
            if i == 0:
                ax.text(
                    -0.3,
                    0.5,
                    f"Ground Truth\nRe={int(re_number)}",
                    fontsize=12,
                    fontweight="bold",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                )

        for model_idx, model_name in enumerate(model_names):
            pred_dict = all_rollouts[model_name]["pred"]

            if (
                var == "vort"
                and "v_x" in pred_dict.keys()
                and "v_y" in pred_dict.keys()
            ):
                pred_dict["vort"] = compute_vorticity(
                    pred_dict["v_x"], pred_dict["v_y"]
                )

            if var not in pred_dict.keys():
                logger.warning(
                    f"Variable '{var}' not in prediction for {model_name}. Skipping."
                )
                continue

            pred_seq = pred_dict[var]

            for i, idx in enumerate(indices[:num_plots]):
                row_idx_pred = 1 + 2 * model_idx
                ax_pred = fig.add_subplot(spec[row_idx_pred, i])
                ax_pred.imshow(pred_seq[idx].cpu().T, cmap=cmap, origin="lower")
                ax_pred.axis("off")
                if i == 0:
                    ax_pred.text(
                        -0.3,
                        0.5,
                        f"{model_name}\nPrediction",
                        fontsize=12,
                        fontweight="bold",
                        transform=ax_pred.transAxes,
                        ha="right",
                        va="center",
                    )

                row_idx_err = 2 + 2 * model_idx
                err = gt_seq[idx] - pred_seq[idx]
                ax_err = fig.add_subplot(spec[row_idx_err, i])
                ax_err.imshow(err.cpu().T, cmap="coolwarm", origin="lower")
                ax_err.axis("off")
                if i == 0:
                    ax_err.text(
                        -0.3,
                        0.5,
                        f"{model_name}\nError",
                        fontsize=12,
                        fontweight="bold",
                        transform=ax_err.transAxes,
                        ha="right",
                        va="center",
                    )

        fig.suptitle(
            f"Joint Rollout Comparison for Variable: {var.upper()}", fontsize=16
        )
        plt.tight_layout(rect=[0.1, 0, 1, 0.95])
        plt.savefig(f"joint_rollout_{var}.png", dpi=300, bbox_inches="tight")
        plt.close()


def generate_comparison_latex_table(
    all_results_by_re: dict,
    caption: str = "Quantitative comparison for different models with Re embedding",
    label: str = "tab:quantitative_comparison",
):
    """
    Generates a LaTeX table formatted exactly like the provided template.

    Args:
        all_results_by_re (dict): A nested dictionary with the evaluation results.
            Expected format:
            {
                RE_VALUE_1: {
                    'ModelName1': {'l2': (mean, std), 'ssim': (mean, std), 'url': '...'},
                    'ModelName2': {'l2': (mean, std), 'ssim': (mean, std), 'url': '...'}
                },
                RE_VALUE_2: { ... }
            }
            'l2' is used for MSE, 'ssim' for LSiM. The 'url' key is optional.
        caption (str): The table's caption.
        label (str): The table's LaTeX label.

    Returns:
        str: A string containing the full LaTeX code for the table.
    """
    if not all_results_by_re:
        return "No results to display."

    # --- Configuration based on the template ---
    METRIC_MAP = {"l2": "MSE", "ssim": "LSiM"}
    METRIC_ORDER = ["MSE", "LSiM"]
    MODEL_DISPLAY_MAP = {
        "ACDM_ncn": r"$ACDM_{ncn}$",
        # Add other special display names here if needed
    }
    # SCALING_MAP format: {Re: {Metric: (scale_factor, precision, latex_exponent)}}
    SCALING_MAP = {
        150: {"MSE": (1e-4, 1, -4), "LSiM": (1e-2, 2, -2)},
        300: {"MSE": (1e-5, 2, -5), "LSiM": (1e-2, 3, -2)},
        500: {"MSE": (1e-6, 1, -6), "LSiM": (1e-3, 1, -3)},
        600: {"MSE": (1e-5, 1, -5), "LSiM": (1e-2, 2, -2)},
        1000: {"MSE": (1e-5, 1, -5), "LSiM": (1e-2, 2, -2)},
    }

    def format_value(mean, std, scale, precision):
        if np.isnan(mean) or np.isnan(std):
            return "-"
        return f"${mean / scale:.{precision}f} \\pm {std / scale:.{precision}f}$"

    # --- Extract sorted keys for consistent ordering ---
    re_values = sorted(list(all_results_by_re.keys()))
    if not re_values:
        return "No Re values found in results."
    model_names = sorted(list(all_results_by_re[re_values[0]].keys()))

    # --- Build LaTeX String ---
    # Preamble and table definition
    latex_parts = [
        r"\begin{table*}[h!]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt} % tighter spacing",
        r"\resizebox{0.9\textwidth}{!}{%",
        r"\begin{tabular}{@{}l " + " ".join(["cc"] * len(re_values)) + "@{}}",
        r"\toprule",
    ]

    # Header Row 1 (Re values)
    header_row1 = [r"\multirow{2}{*}{\textbf{Method}}"]
    header_row1.extend([f"\\multicolumn{{2}}{{c}}{{{re}}}" for re in re_values])
    latex_parts.append(" & ".join(header_row1) + r" \\")

    # Header Row 2 (Metric names and cmidrules)
    cmidrules = []
    for i, _ in enumerate(re_values):
        start_col, end_col = 2 + i * 2, 3 + i * 2
        cmidrules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
    latex_parts.append(" ".join(cmidrules))

    header_row2 = [""]
    for re in re_values:
        for metric in METRIC_ORDER:
            _, _, exponent = SCALING_MAP[re][metric]
            header_row2.append(f"{metric} $(\\times 10^{{{exponent}}})$")
    latex_parts.append(" & ".join(header_row2) + r" \\")
    latex_parts.append(r"\midrule")

    # Data Rows
    for i, model_name in enumerate(model_names):
        display_name = MODEL_DISPLAY_MAP.get(model_name, model_name)
        url = all_results_by_re[re_values[0]][model_name].get("url")
        if url:
            display_name = f"\\href{{{url}}}{{{display_name}}}"

        row_data = [display_name]
        for re in re_values:
            for display_metric in METRIC_ORDER:
                internal_key = next(
                    k for k, v in METRIC_MAP.items() if v == display_metric
                )

                # Get data, handling missing models/metrics gracefully
                metrics_dict = all_results_by_re.get(re, {}).get(model_name, {})
                mean, std = metrics_dict.get(internal_key, (np.nan, np.nan))

                scale, precision, _ = SCALING_MAP[re][display_metric]
                row_data.append(format_value(mean, std, scale, precision))

        latex_parts.append(" & ".join(row_data) + r" \\")
        if i < len(model_names) - 1:
            latex_parts.append(r"\midrule")

    # Footer
    latex_parts.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
        ]
    )

    return "\n".join(latex_parts)


def main(eval_config_path, device, rollout_length, generate_plots):
    """Main function to run the evaluation for multiple models."""
    with open(eval_config_path, "r") as f:
        eval_config = yaml.safe_load(f)

    all_metrics_results = {}
    all_rollout_data = {}

    # --- Create a single, canonical KAE val_loader for all evaluations ---
    kae_config_for_loader = next(
        (m for m in eval_config["evaluations"] if m["model_type"] == "kae"), None
    )
    if not kae_config_for_loader:
        raise ValueError(
            "Evaluation config must contain at least one KAE model to create a canonical validation loader."
        )

    logger.info(
        f"Creating canonical validation loader from KAE config: {kae_config_for_loader['name']}"
    )
    base_config = OmegaConf.structured(Config)
    file_config = OmegaConf.load(kae_config_for_loader["config_path"])
    cfg_loader = OmegaConf.merge(base_config, file_config)
    OmegaConf.resolve(cfg_loader)
    cfg_loader.data.max_sequence_length = rollout_length

    train_ds, val_ds, test_ds = load_datasets(cfg_loader)
    _, canonical_val_loader, canonical_test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, cfg_loader.training
    )

    # --- Loop through and evaluate all models ---
    for model_config in eval_config.get("evaluations", []):
        model_name = model_config.get("name", model_config["model_type"])
        model_type = model_config["model_type"]

        if model_type.lower() == "kae":
            model, cfg = load_kae_model(model_config, device)
            metrics_result = run_full_evaluation_and_report(
                model=model,
                loader=canonical_val_loader,
                input_len=2,
                output_len=rollout_length,
            )
            logger.info(f"Metric results: {metrics_result}")
            if generate_plots:
                logger.info(f"Generating rollout data for {model_name} for plotting...")
                gt, pred, meta = run_rollout_for_plots_kae(
                    model, canonical_val_loader, rollout_length, device
                )

        elif model_type.lower() in ["acdm", "acdm_ncn"]:
            model, acdm_test_loader = load_acdm_model_and_loader(model_config, device)
            metrics_result = run_full_eval_and_report_diffusion(
                model=model,
                loader=[
                    acdm_test_loader,
                    canonical_val_loader,
                ],  # Use ACDM loader for rollout, KAE loader for normalization
                device=device,
                numSamples=1,
                sequenceLength=rollout_length,
            )
            if generate_plots:
                logger.info(f"Generating rollout data for {model_name} for plotting...")
                gt, pred, meta = run_rollout_for_plots_diffusion(
                    model, acdm_test_loader, rollout_length, device
                )

        all_metrics_results[model_name] = metrics_result
        if generate_plots:
            all_rollout_data[model_name] = {
                "gt": gt.cpu(),
                "pred": pred.cpu(),
                "metadata": meta,
            }

    # --- Report Results ---
    comparison_latex_table = generate_comparison_latex_table(all_metrics_results)
    print("\n--- Combined Metrics LaTeX Table ---\n")
    print(comparison_latex_table)

    output_filename = "metrics_table_comparison.tex"
    with open(output_filename, "w") as f:
        f.write(comparison_latex_table)
    logger.info(f"Combined LaTeX table saved to {output_filename}")

    if generate_plots:
        generate_joint_rollout_plots(
            all_rollout_data, variables=["p", "v_x", "v_y", "vort"]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate model evaluation for multiple models."
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        required=True,
        help="Path to the YAML evaluation config file.",
    )
    parser.add_argument(
        "--rollout_length",
        type=int,
        default=60,
        help="Number of steps to roll out the models.",
    )
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        help="Generate and save joint rollout plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation (cuda or cpu).",
    )

    args = parser.parse_args()

    main(
        args.eval_config,
        torch.device(args.device),
        args.rollout_length,
        args.generate_plots,
    )

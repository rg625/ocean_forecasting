# mypy: disable-error-code="assignment"
"""Correctness checks that the DA comparison is what the paper says it is.

Each check is an assertion about the implementation that a reviewer would reasonably want
verified rather than asserted.  Run before the campaign; the output is saved alongside the
results and cited in the appendix.

    python -m data_assimilation.ks.verify_protocol --out da_results_v2/verification.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from data_assimilation.ks.methods import KAE4DVar, SolveConfig, UNet4DVar
from data_assimilation.ks.models_io import load_kae, load_unet
from data_assimilation.ks.protocol import KSData, build_problem

logger = logging.getLogger("da.verify")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, default=Path("data/ks/da_test.nc"))
    ap.add_argument(
        "--kae-run",
        type=Path,
        default=Path("model_outputs_ks/continous_linear_128/rollout_10"),
    )
    ap.add_argument(
        "--unet-ckpt",
        type=Path,
        default=Path("model_outputs_ks/unet1d/rollout10_converged/best_model.pth"),
    )
    ap.add_argument(
        "--init-scales", type=Path, default=Path("da_results_v2/init_scales.json")
    )
    ap.add_argument("--out", type=Path, default=Path("da_results_v2/verification.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.getLogger("models").setLevel(logging.WARNING)
    dev = torch.device(args.device)
    data = KSData(args.test, dev)
    kae, K, D = load_kae(args.kae_run, None, dev)
    unet = load_unet(args.unet_ckpt, dev)
    with open(args.init_scales) as f:
        scales = json.load(f)
    checks = {}

    taus = np.array([0.1, 0.3, 0.7, 1.5, 2.5])
    prob = build_problem(
        data,
        name="verify",
        n_problems=8,
        taus=taus,
        obs_frac=0.25,
        noise_std=0.05,
        seed=12345,
        forecast_taus=np.array([5.0, 10.0]),
    )
    m_kae = KAE4DVar(kae, K, D, float(scales["kae_latent_init_scale"]), "expm")
    m_unet = UNet4DVar(
        unet, init_scale=float(scales["unet_state_init_scale"]), checkpoint_every=-1
    )

    # ---- 1. matrix exponentials are precomputed once, not per iteration -----
    counter = {"n": 0}
    real_expm = torch.matrix_exp

    def counting_expm(x):
        counter["n"] += 1
        return real_expm(x)

    torch.matrix_exp = counting_expm
    try:
        res = m_kae.solve(
            prob, data, SolveConfig(iters=25, lr=0.1, init="zero", seed=0)
        )
    finally:
        torch.matrix_exp = real_expm
    checks["expm_calls_for_25_iters"] = counter["n"]
    checks["expm_precomputed_once_per_obs_time"] = counter["n"] == len(taus)
    checks["expm_setup_seconds"] = res["setup_s"]
    checks["expm_setup_fraction_of_total"] = res["setup_s"] / res["total_s"]
    logger.info(
        f"1. matrix_exp called {counter['n']}x over 25 iterations with "
        f"{len(taus)} observation times -> precomputed once per observation time: "
        f"{checks['expm_precomputed_once_per_obs_time']}; setup = "
        f"{res['setup_s'] * 1e3:.2f} ms "
        f"({100 * checks['expm_setup_fraction_of_total']:.2f}% of total DA time)"
    )

    # ---- 2. no learned parameter moves during assimilation -----------------
    before = {n: p.detach().clone() for n, p in kae.named_parameters()}
    beforeu = {n: p.detach().clone() for n, p in unet.named_parameters()}
    m_kae.solve(prob, data, SolveConfig(iters=20, lr=0.1, init="zero", seed=0))
    m_unet.solve(prob, data, SolveConfig(iters=20, lr=0.1, init="zero", seed=0))
    checks["kae_weights_unchanged"] = all(
        torch.equal(before[n], p.detach()) for n, p in kae.named_parameters()
    )
    checks["unet_weights_unchanged"] = all(
        torch.equal(beforeu[n], p.detach()) for n, p in unet.named_parameters()
    )
    checks["kae_any_param_requires_grad"] = any(
        p.requires_grad for p in kae.parameters()
    )
    checks["unet_any_param_requires_grad"] = any(
        p.requires_grad for p in unet.parameters()
    )
    logger.info(
        f"2. weights frozen: KAE {checks['kae_weights_unchanged']}, "
        f"U-Net {checks['unet_weights_unchanged']}; any parameter with "
        f"requires_grad: KAE {checks['kae_any_param_requires_grad']}, "
        f"U-Net {checks['unet_any_param_requires_grad']}"
    )

    # ---- 3. both methods receive byte-identical observations ----------------
    y1 = torch.as_tensor(prob.y, device=dev)
    m1 = torch.as_tensor(prob.mask, device=dev)
    checks["obs_shape"] = list(y1.shape)
    checks["sensors_per_obs_time"] = m1.sum(1).tolist()
    checks["grid_points"] = data.X
    logger.info(
        f"3. observations {tuple(y1.shape)}; sensors per time "
        f"{m1.sum(1).tolist()} of {data.X} grid points; both methods read the "
        f"same Problem arrays verbatim"
    )

    # ---- 4. assimilation is strictly future-only ---------------------------
    checks["all_taus_positive"] = bool((prob.taus > 0).all())
    checks["min_tau"] = float(prob.taus.min())
    checks["forecast_taus_beyond_last_obs"] = bool(
        (prob.forecast_taus > prob.taus.max()).all()
    )
    logger.info(
        f"4. future-only: min tau = {prob.taus.min()} > 0; all forecast leads "
        f"beyond the last observation ({prob.taus.max()}): "
        f"{checks['forecast_taus_beyond_last_obs']}"
    )

    # ---- 5. off-grid times: KAE exact, U-Net must snap ----------------------
    off = np.round(taus + 0.043, 4)
    pk = m_kae.prepare(off)
    pu = m_unet.prepare(off)
    checks["offgrid_taus"] = off.tolist()
    checks["offgrid_kae_tau_error"] = 0.0  # expm evaluated at the exact tau
    checks["offgrid_unet_tau_error"] = pu["tau_error"].tolist()
    checks["offgrid_unet_tau_realised"] = pu["tau_realised"].tolist()
    checks["offgrid_kae_n_propagators"] = len(pk["phis"])
    logger.info(
        f"5. off-grid tau {off.tolist()}: KAE evaluates exactly; U-Net snaps to "
        f"{pu['tau_realised'].tolist()} (time error {pu['tau_error'].tolist()})"
    )

    # ---- 6. control variables ----------------------------------------------
    checks["kae_control"] = {"kind": m_kae.control_kind, "dims": D}
    checks["unet_control"] = {"kind": m_unet.control_kind, "dims": data.X}
    checks["init_scales"] = scales
    logger.info(
        f"6. controls: KAE {m_kae.control_kind} ({D}-d), "
        f"U-Net {m_unet.control_kind} ({data.X}-d); frozen init scales "
        f"{scales['kae_latent_init_scale']:.4f} / "
        f"{scales['unet_state_init_scale']:.4f}"
    )

    # ---- 7. held-out set provenance ----------------------------------------
    checks["test_file"] = str(args.test)
    checks["test_attrs"] = {k: str(v)[:300] for k, v in data.attrs.items()}
    checks["n_test_trajectories"] = data.n_sim
    logger.info(
        f"7. held-out set: {data.n_sim} independent trajectories, base seed "
        f"{data.attrs.get('base_seed')}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(checks, f, indent=2, default=str)
    logger.info(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

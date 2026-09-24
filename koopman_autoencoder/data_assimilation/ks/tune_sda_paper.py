"""Select the SDA sampler settings on VALIDATION data only.

Tunes, in this order of importance: the number of reverse steps N, the number of Langevin
corrections C, the corrector parameter tau, and the Gamma variant. Also selects k by
comparing trained blanket widths. ``da_test.nc`` is never opened here.

    python data_assimilation/ks/tune_sda_paper.py --out da_results_sda_paper/tuning.json
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

from data_assimilation.ks.protocol import DT, KSData, build_problem
from data_assimilation.ks.sda_paper import SDA, CosineVPSchedule, LocalScoreUNet
from data_assimilation.ks.sda_paper_eval import SamplerConfig, configure, draw, score

logger = logging.getLogger("tune_sda_paper")
CANON = np.array([0.1, 0.3, 0.7, 1.5, 2.5])


def load(ckpt: Path, dev):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = LocalScoreUNet(
        k=int(st["k"]), hidden=tuple(st["hidden"]), blocks=int(st["blocks"])
    ).to(dev)
    net.load_state_dict(st["model_state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    G = torch.load(ckpt.parent / "gamma.pt", map_location=dev, weights_only=False)[
        "Gamma"
    ].to(dev)
    return SDA(net, CosineVPSchedule(), dev, gamma=G), G, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpts",
        nargs="+",
        type=Path,
        default=[
            Path("model_outputs_ks/sda_paper/k2/best_model.pth"),
            Path("model_outputs_ks/sda_paper/k4/best_model.pth"),
        ],
    )
    ap.add_argument("--val", type=Path, default=Path("data/ks/val.nc"))
    ap.add_argument("--L", type=int, default=56)
    ap.add_argument("--n-problems", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument(
        "--out", type=Path, default=Path("da_results_sda_paper/tuning.json")
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    dev = torch.device(args.device)
    data = KSData(args.val, dev)
    prob = build_problem(
        data,
        name="tune",
        n_problems=args.n_problems,
        taus=CANON,
        obs_frac=0.25,
        noise_std=0.05,
        seed=0,
        forecast_taus=np.array([float((args.L - 1) * DT)]),
    )

    rows = []

    def run(tag, sda, G, **kw):
        cfg = SamplerConfig(n_samples=args.n_samples, chunk=args.n_problems, **kw)
        configure(sda, cfg, G)
        r = score(draw(sda, data, prob, args.L, cfg), data, prob, args.L)
        row = {
            "tag": tag,
            **{k: v for k, v in kw.items()},
            "analysis": float(np.mean(r["analysis_rel_l2"])),
            "unobs": float(np.mean(r["unobs_frame_rel_l2"])),
            "obs": float(np.mean(r["obs_rel_l2"])),
            "cov95": r["coverage_95"],
            "spread_err": r["spread_error_ratio"],
            "wall_s": r["wall_s"],
            "seg_evals": r["n_segment_evals"],
        }
        rows.append(row)
        logger.info(
            f"{tag:34s} analysis {row['analysis']:.4f} | unobs {row['unobs']:.4f} "
            f"| obs {row['obs']:.4f} | cov95 {row['cov95']:.3f} "
            f"| {row['wall_s']:.0f}s"
        )
        return row

    # ---- 1. blanket width k, at a fixed reasonable sampler -----------------
    best_k, best_row, models = None, None, {}
    for ck in args.ckpts:
        if not ck.is_file():
            logger.info(f"missing {ck}, skipping")
            continue
        sda, G, st = load(ck, dev)
        models[st["k"]] = (sda, G, ck)
        r = run(
            f"k={st['k']} N=128 C=1 tau=0.5",
            sda,
            G,
            n_steps=128,
            corrections=1,
            tau=0.5,
            gamma_mode="appendix_b",
            gamma_floor=1e-2,
        )
        r["k"] = int(st["k"])
        if best_row is None or r["analysis"] < best_row["analysis"]:
            best_k, best_row = int(st["k"]), r
    logger.info(f"--> selected blanket k = {best_k}")
    sda, G, ck = models[best_k]

    # ---- 2. reverse steps N (convergence) ----------------------------------
    for N in [64, 128, 256, 512]:
        run(
            f"N={N}",
            sda,
            G,
            n_steps=N,
            corrections=1,
            tau=0.5,
            gamma_mode="appendix_b",
            gamma_floor=1e-2,
        )
    N_rows = [
        r for r in rows if r["tag"].startswith("N=") and np.isfinite(r["analysis"])
    ]
    bestN = min(N_rows, key=lambda r: r["analysis"])
    # cheapest N within 5% of the best -> converged and inexpensive
    okN = [r for r in N_rows if r["analysis"] <= 1.05 * bestN["analysis"]]
    N_sel = min(int(r["n_steps"]) for r in okN)
    logger.info(
        f"--> selected N = {N_sel} (best {bestN['n_steps']} at "
        f"{bestN['analysis']:.4f}; cheapest within 5%)"
    )

    # ---- 3. corrections C ---------------------------------------------------
    for C in [0, 1, 2, 4]:
        run(
            f"C={C}",
            sda,
            G,
            n_steps=N_sel,
            corrections=C,
            tau=0.5,
            gamma_mode="appendix_b",
            gamma_floor=1e-2,
        )
    C_sel = int(
        min(
            [
                r
                for r in rows
                if r["tag"].startswith("C=") and np.isfinite(r["analysis"])
            ],
            key=lambda r: r["analysis"],
        )["corrections"]
    )
    logger.info(f"--> selected C = {C_sel}")

    # ---- 4. corrector parameter tau ----------------------------------------
    for tau in [0.1, 0.25, 0.5, 1.0]:
        run(
            f"tau={tau}",
            sda,
            G,
            n_steps=N_sel,
            corrections=max(C_sel, 1),
            tau=tau,
            gamma_mode="appendix_b",
            gamma_floor=1e-2,
        )
    tau_sel = float(
        min(
            [
                r
                for r in rows
                if r["tag"].startswith("tau=") and np.isfinite(r["analysis"])
            ],
            key=lambda r: r["analysis"],
        )["tau"]
    )
    logger.info(f"--> selected tau = {tau_sel}")

    # ---- 5. Gamma variant ---------------------------------------------------
    # The raw Appendix-B Gamma is near-singular for KS (the spatial power spectrum spans
    # ~6 decades), so its eigenvalue floor is itself a tunable numerical regulariser.
    for fl in [1e-3, 1e-2, 1e-1]:
        run(
            f"gamma=appendix_b floor={fl}",
            sda,
            G,
            n_steps=N_sel,
            corrections=C_sel,
            tau=tau_sel,
            gamma_mode="appendix_b",
            gamma_floor=fl,
        )
    for gs in [1e-1, 1e-2, 1e-3]:
        run(
            f"gamma=scalar {gs}",
            sda,
            G,
            n_steps=N_sel,
            corrections=C_sel,
            tau=tau_sel,
            gamma_mode="scalar",
            gamma_scale=gs,
        )
    g_rows = [
        r for r in rows if r["tag"].startswith("gamma=") and np.isfinite(r["analysis"])
    ]
    g_best = min(g_rows, key=lambda r: r["analysis"])
    logger.info(f"--> selected Gamma = {g_best['tag']}")

    sel = {
        "k": best_k,
        "ckpt": str(ck),
        "n_steps": N_sel,
        "corrections": C_sel,
        "tau": tau_sel,
        "gamma_mode": g_best["gamma_mode"],
        "gamma_scale": g_best.get("gamma_scale", 1e-2),
        "gamma_floor": g_best.get("gamma_floor", 1e-2),
        "L": args.L,
        "selected_on": str(args.val),
        "note": (
            "All settings chosen on the validation split; da_test.nc was not "
            "opened by this script."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "selected": sel}, f, indent=2, default=float)
    logger.info(f"selected: {json.dumps(sel)}")
    logger.info(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

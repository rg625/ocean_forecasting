"""The same initialisation choice for the U-Net, decided on validation.

If the KAE is given a background first guess, the U-Net must be offered one too, or the
comparison is not matched. Its control is the state at t_0, so its analogue of the KAE's
e^{-K tau} enc(y_1) is simply y_1 -- the nearest observation, with no backward map available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_assimilation.ks.exp_geometry import schedule
from data_assimilation.ks.methods import SolveConfig, UNet4DVar
from data_assimilation.ks.models_io import load_unet
from data_assimilation.ks.protocol import DT, KSData, build_problem, rel_l2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--unet-ckpt",
        type=Path,
        default=Path("model_outputs_ks/unet1d/rollout10_extended2/best_model.pth"),
    )
    ap.add_argument("--val", type=Path, default=Path("data/ks/val.nc"))
    ap.add_argument(
        "--out", type=Path, default=Path("da_results_geometry_df9/unet_tuning.json")
    )
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 9])
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.03])
    ap.add_argument("--inits", nargs="+", default=["zero", "obs"])
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--n-problems", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    data = KSData(a.val, dev)
    unet = load_unet(a.unet_ckpt, dev)
    rows = []
    for df in a.deltas:
        fr = schedule(df, 25, 5)
        prob = build_problem(
            data, name=f"U{df}", n_problems=a.n_problems, taus=fr * DT, seed=a.seed
        )
        u0 = data.frames(
            torch.as_tensor(prob.sim, device=dev), torch.as_tensor(prob.t0, device=dev)
        )
        i = int(np.argmin(prob.taus))
        y1 = torch.as_tensor(prob.y[i], device=dev)  # nearest observation
        for init in a.inits:
            ik = (
                {"init": "given", "init_value": y1} if init == "obs" else {"init": init}
            )
            for lr in a.lrs:
                m = UNet4DVar(unet, init_scale=1.0, checkpoint_every=-1)
                r = m.solve(
                    prob, data, SolveConfig(iters=a.iters, lr=lr, seed=a.seed, **ik)
                )
                with torch.no_grad():
                    v = (
                        rel_l2(
                            data.denorm(m.analysis_field(r["control"])), data.denorm(u0)
                        )
                        .cpu()
                        .numpy()
                    )
                rows.append(
                    {
                        "delta_f": df,
                        "init": init,
                        "lr": lr,
                        "mean": float(v.mean()),
                        "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
                    }
                )
                print(
                    f"val U-Net df={df:2d} init={init:5s} lr={lr:<6g} {v.mean():.4f}",
                    flush=True,
                )
                a.out.parent.mkdir(parents=True, exist_ok=True)
                a.out.write_text(
                    json.dumps(
                        {
                            "meta": {
                                "val": str(a.val),
                                "iters": a.iters,
                                "n_problems": a.n_problems,
                            },
                            "grid": rows,
                        },
                        indent=2,
                    )
                )
    best, score = None, np.inf
    for init in a.inits:
        for lr in a.lrs:
            sel = [r["mean"] for r in rows if r["init"] == init and r["lr"] == lr]
            s = float(np.mean(np.log(sel)))
            if s < score:
                best, score = {"init": init, "lr": lr}, s
    doc = json.loads(a.out.read_text())
    doc["best"] = best
    a.out.write_text(json.dumps(doc, indent=2))
    print("BEST (frozen):", best)


if __name__ == "__main__":
    main()

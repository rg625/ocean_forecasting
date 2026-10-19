"""Evaluate a CT-KAE checkpoint on the TRA benchmark with the unified protocol.

    python eval_ctkae.py RUN_DIR [--ckpt best.pth] [--save_npz] [--dt_test] [--ma_sweep]

1. Benchmark cases (gt_interp / gt_extrap / gt_longer, frames 0.1 apart): input = frames
   0,1, prediction = frames 2..T-1 from ONE exact latent propagation. The ground truth
   rebuilt from the .nc files is asserted equal to the ACDM groundTruth.dict, then
   scored with audit_report/scripts/benchmark_suite.score (same code as every baseline).
2. --dt_test: zero-shot temporal resolution on gt_<case>_subsample_1.nc (frames 0.05
   apart): the input window keeps the training spacing (frames 0 and 2), the model is
   queried at every 0.05 lead; errors at half-step leads (never trained on when
   strides=[2]) are reported separately from whole-step leads.
3. --ma_sweep: generator spectrum (block: g_j, w_j) and output sensitivity vs Mach.
Results: RUN_DIR/eval/<ckpt>.json (+ npz in ACDM layout, fp16, with --save_npz).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "audit_report" / "scripts"))
from benchmark_suite import load_gt, score, train_climatology  # noqa: E402
from models.ct_data import load_eval_case, to_acdm_layout  # noqa: E402
from models.ct_kae import CTKAE, CTKAEConfig  # noqa: E402

DATA = HERE / "data/acdm/128_tra"


def load_model(run_dir: Path, ckpt: str, dev):
    ck = torch.load(run_dir / ckpt, map_location="cpu", weights_only=False)
    m = CTKAE(CTKAEConfig(**ck["model_config"]))
    m.load_state_dict(ck["model"], strict=True)
    return m.to(dev).eval(), ck


@torch.no_grad()
def rollout(m, x, ma, mask, n_in_stride: int, taus: torch.Tensor, chunk: int = 60):
    """x: [S, T, C, H, W]; window = frames ending at index (n-1)*stride with spacing stride."""
    n = m.cfg.n_frames
    win = x[:, 0 : (n - 1) * n_in_stride + 1 : n_in_stride]
    z0 = m.encode(win, mask, ma)
    zs = m.propagate(z0, taus.expand(x.shape[0], -1), ma)
    outs = []
    for s in range(x.shape[0]):
        parts = []
        for k in range(0, zs.shape[1], chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                parts.append(
                    m.decode(
                        zs[s, k : k + chunk],
                        mask,
                        ma[s : s + 1].expand(min(chunk, zs.shape[1] - k)),
                    ).float()
                )
        outs.append(torch.cat(parts))
    return torch.stack(outs)  # [S, K, C, H, W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="best.pth")
    ap.add_argument("--save_npz", action="store_true")
    ap.add_argument("--dt_test", action="store_true")
    ap.add_argument("--ma_sweep", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    run = Path(args.run_dir)
    m, ck = load_model(run, args.ckpt, dev)
    out = {"ckpt": str(run / args.ckpt), "step": ck.get("step"), "val": ck.get("val")}
    clim = train_climatology()
    (run / "eval").mkdir(exist_ok=True)
    for case in ["interp", "extrap", "longer"]:
        x, ma, mask = load_eval_case(str(DATA / f"gt_{case}.nc"), dev)
        S, T = x.shape[:2]
        taus = 0.1 * torch.arange(1, T - 1, device=dev, dtype=torch.float32)
        pred = rollout(m, x, ma, mask, 1, taus)  # frames 2..T-1
        full = torch.cat([x[:, :2], pred], 1)  # frames 0,1 = inputs (ACDM convention)
        arr = to_acdm_layout(full, ma)[:, :, :4]
        gt, fluid = load_gt(case)
        gt_mine = to_acdm_layout(x, ma)[:, :, :4] * fluid
        gdiff = float(np.abs(gt_mine - gt).max())
        assert (
            gdiff < 1e-4
        ), f"{case}: rebuilt ground truth differs from groundTruth.dict ({gdiff})"
        r = score(arr[None], gt, fluid, clim)
        r["gt_rebuild_maxdiff"] = gdiff
        out[case] = r
        print(
            f"{case:7s} mse {r['mse_pred']:.3f}  acc<0.6@{r['first_acc_below_0.6']}  E {r['energy_ratio_at']}"
        )
        if args.save_npz:
            ma5 = to_acdm_layout(full, ma)
            np.savez_compressed(
                run / "eval" / f"{case}.npz", arr_0=ma5[None, None].astype(np.float16)
            )
    if args.dt_test:
        out["dt_test"] = {}
        for case in ["interp", "extrap"]:
            p = DATA / f"gt_{case}_subsample_1.nc"
            x, ma, mask = load_eval_case(str(p), dev)  # frames 0.05 apart
            T = x.shape[1]
            taus = 0.05 * torch.arange(
                1, T - 2, device=dev, dtype=torch.float32
            )  # from frame 2
            pred = rollout(m, x, ma, mask, 2, taus)  # window = frames 0, 2
            tgt = x[:, 3:T]
            from models.ct_data import denormalise

            e = ((denormalise(pred) - denormalise(tgt)) ** 2).mean(
                dim=(0, 2, 3, 4)
            ) * 1e3  # [K]
            k = torch.arange(1, e.shape[0] + 1)
            half, whole = e[k % 2 == 1], e[k % 2 == 0]
            out["dt_test"][case] = {
                "mse_half_step_leads": float(half.mean()),
                "mse_whole_step_leads": float(whole.mean()),
                "mse_lead": [float(v) for v in e],
            }
            print(
                f"dt-test {case}: half-step leads {float(half.mean()):.3f}  whole-step leads {float(whole.mean()):.3f}"
            )
    if args.ma_sweep:
        grid = torch.linspace(0.45, 0.95, 51, device=dev)
        with torch.no_grad():
            e = m.embed_cond(grid)
            res = {"ma": grid.tolist()}
            if m.cfg.generator == "block":
                g, w = m.gen.params(e, len(grid))
                res["gamma"] = g.T.tolist()
                res["omega"] = w.T.tolist()
            else:
                ev = torch.linalg.eigvals(m.gen.matrix(e, len(grid)))
                res["spectral_abscissa"] = ev.real.max(-1).values.tolist()
            x, ma, mask = load_eval_case(str(DATA / "gt_interp.nc"), dev)
            taus = 0.1 * torch.arange(1, 59, device=dev, dtype=torch.float32)
            base = rollout(m, x[:1], ma[:1], mask, 1, taus)
            sens = []
            for v in [0.50, 0.55, 0.60, 0.66, 0.70, 0.80, 0.90]:
                p = rollout(m, x[:1], torch.full((1,), v, device=dev), mask, 1, taus)
                sens.append([v, float(((p - base) ** 2).mean().sqrt())])
            res["output_rms_change_vs_true_ma"] = sens
        out["ma_sweep"] = res
        print(
            "Ma sensitivity (rms change vs true Ma):",
            [(a, round(b, 4)) for a, b in sens],
        )
    (run / "eval" / f"{Path(args.ckpt).stem}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()

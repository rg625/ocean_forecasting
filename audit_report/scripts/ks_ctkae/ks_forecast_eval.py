"""KS forecasting and generator evaluation of the CT-KAE candidates against the paper's models.

On data/ks/test.nc (10 trajectories; 9 starts each, t0 = 100, 200, ..., 900 where the lead fits),
from the exact true state:
  rel_lead     relative L2 error (physical units) at leads 1, 10, 22 (0.1 T_L), 50, 100, 226 (1 T_L)
  mse100       MSE (normalised units) over a 100-step rollout
  roundtrip    encode-decode relative error at t0 (the KAEs' representation floor)
  spectrum     band log10 power ratio (low k <= 3, mid 4-6, high 7-10) over leads 1-100
Generator: eigenvalues of the latent generator (max real part, number with real part > 0) against
the measured leading Lyapunov exponent (da_results_sda_paper/lyapunov.json).
Models: the legacy KS KAE, the U-Net, and every CT-KAE run given.

    python ks_forecast_eval.py OUT.json NAME=RUN_DIR [...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).parent))
import ks_driver  # noqa: E402
from data_assimilation.ks.models_io import load_unet  # noqa: E402
from data_assimilation.ks.protocol import KSData  # noqa: E402

dev = torch.device("cuda")
LEADS = [1, 10, 22, 50, 100, 226]
BANDS = {"low": (1, 3), "mid": (4, 6), "high": (7, 10)}
LEGACY = Path("model_outputs_ks/continous_linear_128/rollout_10")
UNET = Path("model_outputs_ks/unet1d/rollout10_extended2/best_model.pth")


def band_logs(gt, p):
    sg = (torch.fft.rfft(gt.double(), dim=-1).abs() ** 2).mean((0, 1))
    sp = (torch.fft.rfft(p.double(), dim=-1).abs() ** 2).mean((0, 1))
    return {
        b: float(torch.log10(sp[lo : hi + 1] / sg[lo : hi + 1]).mean())
        for b, (lo, hi) in BANDS.items()
    }


def main():
    out = Path(sys.argv[1])
    runs = {
        "legacy KAE": LEGACY,
        **{a.split("=", 1)[0]: Path(a.split("=", 1)[1]) for a in sys.argv[2:]},
    }
    data = KSData(Path("data/ks/test.nc"), dev)
    n = max(LEADS)
    starts = [t for t in range(100, 1000, 100) if t + n < data.n_t]
    sim = torch.arange(data.n_sim, device=dev).repeat_interleave(len(starts))
    t0 = torch.tensor(starts, device=dev).repeat(data.n_sim)
    x0 = data.frames(sim, t0)
    gt = torch.stack(
        [data.frames(sim, t0 + k) for k in range(1, n + 1)]
    )  # [F, B, X] normalised
    gtp = data.denorm(gt)
    lam1 = json.loads(Path("da_results_sda_paper/lyapunov.json").read_text())[
        "lambda_1_mean"
    ]

    def metrics(p, rt=None):
        pp = data.denorm(p)
        rel = (((pp - gtp) ** 2).sum(-1) / (gtp**2).sum(-1)).sqrt().mean(1)  # [F]
        r = {
            "rel_lead": {str(k): float(rel[k - 1]) for k in LEADS},
            "mse100": float(((p[:100] - gt[:100]) ** 2).mean()),
            "spectrum_1_100": band_logs(gt[:100], p[:100]),
        }
        if rt is not None:
            r["roundtrip"] = rt
        return r

    res: dict = {"n_windows": len(x0), "leads": LEADS, "lambda_1": lam1, "models": {}}
    unet = load_unet(UNET, dev)
    with torch.no_grad():
        x, outs = x0, []
        for _ in range(n):
            x = unet(x.unsqueeze(1)).squeeze(1)
            outs.append(x)
    res["models"]["U-Net"] = metrics(torch.stack(outs))
    for name, r in runs.items():
        kae, A, D = ks_driver.load_kae(r, None, dev)
        with torch.no_grad():
            z0 = kae.present_encoding(
                TensorDict({"u": x0[:, None, :, None]}, batch_size=[len(x0), 1]), None
            )
            rec = data.denorm(kae.decode(z0)["u"][..., 0])
            rt = float(
                (
                    ((rec - data.denorm(x0)) ** 2).sum(-1)
                    / (data.denorm(x0) ** 2).sum(-1)
                )
                .sqrt()
                .mean()
            )
            P = torch.matrix_exp(A * 0.1)
            z, outs = z0, []
            for _ in range(n):
                z = z @ P.T
                outs.append(kae.decode(z)["u"][..., 0])
        m = metrics(torch.stack(outs), rt)
        ev = np.linalg.eigvals(A.double().cpu().numpy())
        m["generator"] = {
            "latent_dim": int(D),
            "real_max": float(ev.real.max()),
            "n_real_gt_0": int((ev.real > 0).sum()),
            "real_min": float(ev.real.min()),
            "imag_abs_max": float(np.abs(ev.imag).max()),
            "real_max_over_lambda1": float(ev.real.max() / lam1),
        }
        res["models"][name] = m
        print(
            name,
            json.dumps({k: m[k] for k in ("rel_lead", "mse100", "roundtrip")}),
            m["generator"],
            flush=True,
        )
    print("U-Net", json.dumps(res["models"]["U-Net"]["rel_lead"]))
    out.write_text(json.dumps(res, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()

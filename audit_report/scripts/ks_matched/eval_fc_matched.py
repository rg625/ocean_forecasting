"""Free-running forecast from the true state for the matched KS runs (train_matched.py),
with the paper's FC protocol (data_assimilation/ks/exp_long_horizon.exp_forecast):
da_test_long.nc, trajectories 0..31 from frame t0 = 100, the same lead frames, the same
saturation level and skill-horizon rule (first lead whose mean error reaches 90% of
saturation), exact propagation (KAE: z0 @ expm(K tau)^T as KAE4DVar 'expm'; CT-KAE:
exp(tau A)).

Two encodings of the true state, for every model:
  single : the paper's FC encoding, present_encoding(x_t0) (the CT-KAE delay encoder gets
           x_t0 repeated over its window);
  window : the training encoding, encode(x_{t0-3..t0}) (KAE: (history + present) / 2;
           CT-KAE: the 4-frame delay window).

    python eval_fc_matched.py OUT.npz NAME=RUN_DIR [NAME=RUN_DIR ...] [--ckpt best|final]
RUN_DIR is the run directory holding final_model.pth and checkpoints/.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

MAIN = Path("/home/rg625/mnt/ocean_forecasting/koopman_autoencoder")
IMPL = Path(__file__).resolve().parents[3] / "koopman_autoencoder"
os.chdir(MAIN)
sys.path.insert(0, str(MAIN))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import models  # noqa: E402,F401

_load("models.ct_kae", IMPL / "models/ct_kae.py")
_load("models.ct_kae_ks", IMPL / "models/ct_kae_ks.py")
CM = _load("models.ct_kae_matched", IMPL / "models/ct_kae_matched.py")
import data_assimilation.ks.models_io as MIO  # noqa: E402
from data_assimilation.ks.protocol import DT, KSData, rel_l2  # noqa: E402

N_FRAMES = 4  # the paper config's input_sequence_length


def load(run_dir: Path, which: str, dev):
    """(kind, model, K or None); kind 'kae' | 'ct'"""
    meta = json.loads((run_dir.parent / "matched_run.json").read_text())
    ck = run_dir / (
        "checkpoints/best_model.pth" if which == "best" else "final_model.pth"
    )
    if meta["arch"] == "kae":
        m, K, _ = MIO.load_kae(run_dir, ck, dev)
        return "kae", m, K
    cfg = meta["config"]
    m = CM.CTKAEMatched(
        cfg["data"]["variables"],
        cfg["data"]["input_sequence_length"],
        cfg["model"]["latent_dim"],
        cfg["model"]["hidden_dims"],
    )
    m.load_state_dict(
        torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
    )
    m = m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return "ct", m, None


@torch.no_grad()
def forecast(kind, m, K, z0, taus):
    """[F, B, X] normalised"""
    if kind == "kae":
        return torch.stack(
            [
                m.decode(z0 @ torch.matrix_exp(K * float(t)).T)["u"].squeeze(-1)
                for t in taus
            ]
        )
    B = z0.shape[0]
    out = []
    for i in range(0, len(taus), 32):
        tt = torch.as_tensor(taus[i : i + 32], device=z0.device, dtype=torch.float32)
        zs = m.ct.propagate(z0, tt.expand(B, -1), None)
        out.append(m.decode(zs)["u"].squeeze(-1).transpose(0, 1))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ckpt", choices=["best", "final"], default="best")
    ap.add_argument("--t0-min", type=int, default=100)
    ap.add_argument("--fc-frames", type=int, default=4800)
    ap.add_argument("--fc-problems", type=int, default=32)
    a = ap.parse_args()
    dev = torch.device("cuda")
    TL = float(
        json.loads(Path("da_results_sda_paper/lyapunov.json").read_text())[
            "lyapunov_time"
        ]
    )
    data = KSData(Path("data/ks/da_test_long.nc"), dev)
    n_max, n_prob = a.fc_frames, min(a.fc_problems, data.n_sim)
    rng = np.random.default_rng(7)
    sim = np.arange(n_prob)
    sim_t = torch.as_tensor(sim, device=dev)
    t0_t = torch.full((n_prob,), a.t0_min, device=dev)
    frames = np.unique(
        np.concatenate(
            [np.arange(1, 51), np.round(np.geomspace(51, n_max, 160)).astype(int)]
        )
    ).astype(int)
    taus = frames * DT
    truth_dn = data.denorm(
        torch.stack([data.frames(sim_t, t0_t + int(f)) for f in frames])
    )
    x0 = data.frames(sim_t, t0_t)
    win = torch.stack(
        [data.frames(sim_t, t0_t - k) for k in range(N_FRAMES - 1, -1, -1)], 1
    )  # [B, 4, X]
    oth_s = torch.as_tensor((sim + n_prob // 2) % data.n_sim, device=dev)
    oth_t = torch.as_tensor(rng.integers(a.t0_min, data.n_t - 1, n_prob), device=dev)
    sat = (
        rel_l2(
            data.denorm(data.frames(oth_s, oth_t)).unsqueeze(0).expand_as(truth_dn),
            truth_dn,
        )
        .mean(1)
        .cpu()
        .numpy()
    )
    store = {
        "frames": frames,
        "taus": taus,
        "T_L": np.array(TL),
        "saturation": sat,
        "ckpt": np.array(a.ckpt),
    }
    summary: dict = {}
    for spec in a.runs:
        name, rd = spec.split("=", 1)
        kind, m, K = load(Path(rd), a.ckpt, dev)
        summary[name] = {}
        for enc in ("single", "window"):
            with torch.no_grad():
                if enc == "single":
                    z0 = m.present_encoding(
                        TensorDict({"u": x0[:, None, :, None]}, batch_size=[n_prob, 1]),
                        None,
                    )
                else:
                    z0 = m.encode(
                        TensorDict(
                            {"u": win[..., None]}, batch_size=[n_prob, N_FRAMES]
                        ),
                        None,
                    )
                rt = float(
                    rel_l2(
                        data.denorm(m.decode(z0)["u"].squeeze(-1)), data.denorm(x0)
                    ).mean()
                )
                e = rel_l2(
                    data.denorm(forecast(kind, m, K, z0, taus)), truth_dn
                )  # [F, B]
            mu = e.mean(1).cpu().numpy()
            bad = np.where(mu >= 0.9 * sat)[0]
            h = float(frames[bad[0]] * DT) if len(bad) else float("inf")
            store[f"{name}__{enc}__rel_mean"] = mu
            store[f"{name}__{enc}__rel_sem"] = (
                (e.std(1) / np.sqrt(n_prob)).cpu().numpy()
            )
            at = {
                str(p): float(np.interp(p * TL, taus, mu))
                for p in (0.1, 0.5, 1.0, 2.0, 5.0)
            }
            summary[name][enc] = {
                "roundtrip": rt,
                "rel_at_TL": at,
                "skill_horizon_TL": h / TL,
            }
            print(
                f"{name:22s} {enc:6s} rt {rt:.4f} | "
                + " ".join(f"{k}TL {v:.4f}" for k, v in at.items())
                + f" | horizon {h / TL:.2f} T_L",
                flush=True,
            )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **store)
    a.out.with_suffix(".json").write_text(
        json.dumps({"ckpt": a.ckpt, "T_L": TL, "models": summary}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()

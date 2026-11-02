"""Run the paper's KS experiments with a CT-KAE (models/ct_kae_ks.py) in the KAE slot.

The experiment code is the main repository's data_assimilation/ks, imported unmodified and run
from its directory (so every default path, seed and problem is the paper's); only load_kae is
replaced: a run directory holding last.pth (and no final_model.pth) is loaded as a CT-KAE and
wrapped in the legacy KAE interface (present_encoding, decode(z)["u"], koopman_operator).
Control-variable transform: the CT-KAE latents are unnormalised (per-element std ~350 on the
training split, against ~1 for the paper's KAE, for which the 4D-Var learning-rate grid and init
scale were designed), so the wrapper exposes zeta = S^-1 z with S = diag(per-dimension std of
the encoder latents on the TRAINING split, frames as models_io.kae_latent_scale). The generator
handed to the 4D-Var methods is A' = S^-1 A S, so e^{A' tau} = S^-1 e^{A tau} S is still the
model's own exact propagator. Outputs go wherever --out / --out-dir point (this
repository); nothing is written to the main repository.

    python ks_driver.py init_scales RUN_DIR OUT.json
    python ks_driver.py tune RUN_DIR INIT_SCALES.json OUT.json      # KAE grid of tune.py
    python ks_driver.py run MODULE [module args ...]                # e.g. exp_geometry --kae-run RUN_DIR ...
        env KS_ONLY_KAE=1 restricts da_ks_experiments_3way.ALL / FOURDVAR to KAE-expm (for
        sections without a --methods flag); the other methods' paper results are reused.
"""

import importlib.util
import json
import logging
import os
import runpy
import sys
from pathlib import Path

import torch

MAIN = Path("/home/rg625/mnt/ocean_forecasting/koopman_autoencoder")
IMPL = Path(__file__).resolve().parents[3] / "koopman_autoencoder"
os.chdir(MAIN)
sys.path.insert(0, str(MAIN))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import models  # noqa: E402,F401  (the main repository's package)

_load("models.ct_kae", IMPL / "models/ct_kae.py")
CK = _load("models.ct_kae_ks", IMPL / "models/ct_kae_ks.py")
CMT = _load("models.ct_kae_matched", IMPL / "models/ct_kae_matched.py")
import data_assimilation.ks.models_io as MIO  # noqa: E402

_orig_load_kae = MIO.load_kae


class CTKAEKS(torch.nn.Module):
    """The legacy KAE interface over a CTKAE1d (fixed point 0, no condition), in the
    normalised control variable zeta = S^-1 z."""

    def __init__(self, model, scale):
        super().__init__()
        self.m = model
        self.n = int(model.cfg.n_frames)  # delay window: the present frame repeated
        with torch.no_grad():
            A = model.generator_matrix(None)[0].float()
            self.register_buffer("s", scale.float())
            self.register_buffer("A", A * scale[None, :] / scale[:, None])  # S^-1 A S

    def present_encoding(self, x, cond_input=None):
        u = x["u"][:, -1, :, 0]  # [B, X], the present frame
        w = u[:, None, None, None, :].expand(-1, self.n, -1, -1, -1)
        return self.m.encode(w, None, None) / self.s

    def decode(self, z):
        x = self.m.decode(z * self.s, None, None)  # [..., 1, 1, X]
        return {"u": x.reshape(*z.shape[:-1], x.shape[-1], 1)}

    def koopman_operator(self, z, cond=None, dt=0.1):
        """one classical RK4 step of dz/dt = A z (the 'rk4' propagator of the campaign)"""

        def f(v):
            return v @ self.A.T

        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        return z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def latent_std(model, device):
    """per-dimension std of the encoder latents on the training split (frames as
    models_io.kae_latent_scale: 8 simulations, 512 times from frame 100)"""
    from data_assimilation.ks.protocol import KSData

    data = KSData(Path("data/ks/train.nc"), device)
    idx_t = torch.linspace(100, data.n_t - 1, min(512, data.n_t - 100)).long()
    f = data.u[torch.arange(min(8, data.n_sim))][:, idx_t].reshape(-1, data.X)
    z = torch.cat(
        [
            model.encode(
                f[i : i + 256][:, None, None, None, :].expand(
                    -1, int(model.cfg.n_frames), -1, -1, -1
                ),
                None,
                None,
            )
            for i in range(0, len(f), 256)
        ]
    )
    return z.std(0)


def load_kae(run_dir, ckpt, device):
    run_dir = Path(run_dir)
    meta = run_dir.parent / "matched_run.json"
    if meta.exists() and json.loads(meta.read_text())["arch"] == "ct":
        # train_matched.py run: CTKAEMatched, best-validation checkpoint as the paper's KAE
        cfg = json.loads(meta.read_text())["config"]
        mm = CMT.CTKAEMatched(
            cfg["data"]["variables"],
            cfg["data"]["input_sequence_length"],
            cfg["model"]["latent_dim"],
            cfg["model"]["hidden_dims"],
        )
        ck = ckpt or run_dir / "checkpoints" / "best_model.pth"
        sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
        mm.load_state_dict(sd)
        model = mm.ct.to(device).eval()
        w = CTKAEKS(model, latent_std(model, device)).to(device).eval()
        for p in w.parameters():
            p.requires_grad_(False)
        return w, w.A, int(w.A.shape[0])
    if not (
        (run_dir / "last.pth").exists() and not (run_dir / "final_model.pth").exists()
    ):
        return _orig_load_kae(run_dir, ckpt, device)
    st = torch.load(
        ckpt or run_dir / "last.pth", map_location="cpu", weights_only=False
    )
    model = CK.CTKAE1d(CK.CTKAEConfig(**st["model_config"]))
    model.load_state_dict(st["model"])
    model = model.to(device).eval()
    w = CTKAEKS(model, latent_std(model, device)).to(device).eval()
    for p in w.parameters():
        p.requires_grad_(False)
    return w, w.A, int(w.A.shape[0])


MIO.load_kae = load_kae


def init_scales(run_dir, out):
    from data_assimilation.ks.protocol import KSData

    dev = torch.device("cuda")
    kae, _, _ = load_kae(run_dir, None, dev)
    data = KSData(Path("data/ks/train.nc"), dev)
    with torch.no_grad():
        s = MIO.kae_latent_scale(kae, data)
    paper = json.loads(Path("da_results_v2/init_scales.json").read_text())
    d = {
        "source": "data/ks/train.nc",
        "kae_run": str(run_dir),
        "kae_latent_init_scale": s,
        "unet_state_init_scale": paper["unet_state_init_scale"],
        "note": "kae_latent_init_scale: per-element std of the CT-KAE encoder latents on the "
        "TRAINING split (data_assimilation.ks.models_io.kae_latent_scale); U-Net scale copied "
        "from da_results_v2/init_scales.json.",
    }
    Path(out).write_text(json.dumps(d, indent=2))
    print(json.dumps(d))


def tune(run_dir, scales_path, out):
    """tune.py's validation grid (24 problems, sparse + noisy canonical schedule, 2000
    iterations, lr x init), for the KAE only; the U-Net entry is the paper's, verbatim. The init
    grid is tune.py's (random, zero) plus the harness's informed first guess 'obs'
    (init_utils: e^{-A tau_1} E(y_1)): zero is the latent origin, the paper KAE's latent mean
    but far from the CT-KAE's latents."""
    from data_assimilation.ks import tune as T
    from data_assimilation.ks.evaluate import evaluate_solution, summarize
    from data_assimilation.ks.init_utils import init_kwargs
    from data_assimilation.ks.methods import KAE4DVar, SolveConfig
    from data_assimilation.ks.protocol import KSData, build_problem

    dev = torch.device("cuda")
    data = KSData(Path("data/ks/val.nc"), dev)
    kae, K, D = load_kae(run_dir, None, dev)
    scales = json.loads(Path(scales_path).read_text())
    m = KAE4DVar(kae, K, D, float(scales["kae_latent_init_scale"]), "expm")
    prob = build_problem(
        data,
        name="tune",
        n_problems=24,
        taus=T.CANONICAL_TAUS,
        obs_frac=0.25,
        noise_std=0.05,
        seed=0,
    )
    rows = []
    for init in list(T.GRID_INIT) + ["obs"]:
        ik = init_kwargs(init, kae, K, prob, dev)
        for lr in T.GRID_LR:
            res = m.solve(prob, data, SolveConfig(iters=2000, lr=lr, seed=0, **ik))
            ev = evaluate_solution(m, res["control"], prob, data)
            s, so = summarize(ev["init_rel_l2"]), summarize(ev["obs_rel_l2"])
            rows.append(
                {
                    "lr": lr,
                    "init": init,
                    "init_rel_l2_mean": s["mean"],
                    "init_rel_l2_median": s["median"],
                    "obs_rel_l2_mean": so["mean"],
                    "s_per_solve": res["total_s"],
                }
            )
            print(
                f"init={init:6s} lr={lr:<6g} analysis={s['mean']:.4f} obs={so['mean']:.4f}",
                flush=True,
            )
    best = min(rows, key=lambda r: r["init_rel_l2_mean"])
    paper = json.loads(Path("da_results_v2/tuning.json").read_text())
    d = dict(paper)
    d.update(
        kae_run=str(run_dir),
        init_scales=scales,
        results={
            "KAE-expm": {"grid": rows, "best": best},
            "UNet-4DVar": paper["results"]["UNet-4DVar"],
        },
        note="KAE-expm re-tuned for this CT-KAE with tune.py's grid and problems; UNet-4DVar copied.",
    )
    Path(out).write_text(json.dumps(d, indent=2))
    print("best", best)


def run(module, args):
    # exp_long_horizon's PF section calls Bench.cfg without the problem, which the informed
    # first guess (init 'obs') needs; fall back to the problem built last (PF builds its
    # problem immediately before solving it)
    import data_assimilation.ks.da_ks_experiments_3way as X3
    import data_assimilation.ks.protocol as PR

    _bp, _cfg = PR.build_problem, X3.Bench.cfg
    last: dict = {}

    def build_problem(*a, **k):
        last["p"] = _bp(*a, **k)
        return last["p"]

    def cfg(self, name, iters, seed, prob=None):
        return _cfg(
            self, name, iters, seed, prob if prob is not None else last.get("p")
        )

    PR.build_problem, X3.Bench.cfg = build_problem, cfg
    if os.environ.get("KS_ONLY_KAE"):
        import data_assimilation.ks.da_ks_experiments_3way as X

        X.ALL[:] = ["KAE-expm"]
        X.FOURDVAR[:] = ["KAE-expm"]
    sys.argv = [module] + args
    runpy.run_module(
        f"data_assimilation.ks.{module}", run_name="__main__", alter_sys=True
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "init_scales":
        init_scales(Path(rest[0]), rest[1])
    elif cmd == "tune":
        tune(Path(rest[0]), rest[1], rest[2])
    elif cmd == "run":
        run(rest[0], rest[1:])
    else:
        raise SystemExit(__doc__)

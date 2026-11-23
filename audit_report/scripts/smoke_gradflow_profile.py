"""Phase I/VI smoke test: forward/backward, gradient flow, checkpoint round trip, and
step-time profiling of the TRA continuous KAE exactly as `Trainer._run_one_epoch` runs it.

Points the model code at --repo (default: the main checkout, read-only), so the same
script can be re-run against the audit branch after fixes.

Usage:
    python smoke_gradflow_profile.py [--repo PATH] [--steps 20] [--json out.json]
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument(
    "--repo", default="/home/rg625/mnt/ocean_forecasting/koopman_autoencoder"
)
ap.add_argument(
    "--data",
    default="/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/data/acdm/128_tra/val.nc",
)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--json", default=None)
ap.add_argument(
    "--scratch",
    default="/tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/563da406-8943-47fe-a8f7-28eddbd027a5/scratchpad",
)
args = ap.parse_args()
sys.path.insert(0, args.repo)

from tensordict import TensorDict  # noqa: E402
from models.autoencoder import KoopmanAutoencoder  # noqa: E402
from models.dataloader import (  # noqa: E402
    QGDatasetMultiSim,
    MeanStdNormalizer,
    custom_collate_fn,
)
from models.loss import KoopmanLoss  # noqa: E402
from models.networks import TransformerConfig  # noqa: E402
from models.trainer import Trainer  # noqa: E402

cfgdir = Path(args.repo) / "configs"
mcfg = OmegaConf.load(cfgdir / "model/continous_linear_128_tra.yaml")
dcfg = OmegaConf.load(cfgdir / "data/tra.yaml")
lcfg = OmegaConf.load(cfgdir / "loss/tra.yaml")
dev = torch.device("cuda")
torch.manual_seed(0)

ds = QGDatasetMultiSim(
    data_path=args.data,
    normalizer=MeanStdNormalizer(),
    input_sequence_length=dcfg.input_sequence_length,
    max_sequence_length=dcfg.max_sequence_length,
    variables=dict(dcfg.variables),
    static_variables=dict(dcfg.static_variables),
    subsample=dcfg.subsample,
    exhaustive=True,
    control_parameters=list(dcfg.control_parameters),
    selection_param=dcfg.selection_param,
)
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[: args.batch * 4].tolist()
batches = [
    custom_collate_fn([ds[i] for i in idx[k * args.batch : (k + 1) * args.batch]])
    for k in range(4)
]


def build():
    return KoopmanAutoencoder(
        data_variables=dict(dcfg.variables),
        input_frames=dcfg.input_sequence_length,
        height=mcfg.height,
        width=mcfg.width,
        latent_dim=mcfg.latent_dim,
        cond_embedding_dim=mcfg.cond_embedding_dim,
        cond_type=mcfg.cond_type,
        operator_mode=mcfg.operator_mode,
        hidden_dims=list(mcfg.hidden_dims),
        transformer_config=TransformerConfig(**mcfg.transformer),
        use_checkpoint=False,
        predict_cond=mcfg.predict_cond,
        cond_grad_enabled=mcfg.cond_grad_enabled,
        disturb_std=mcfg.disturb_std,
        is_continuous=mcfg.is_continuous,
        operator_type=mcfg.operator_type,
        rank=mcfg.rank,
        cond_expansion_type=dcfg.selection_param,
        use_attention=mcfg.use_attention,
        spectral=mcfg.spectral,
        **mcfg.conv_kwargs,
    ).to(dev)


model = build()
crit = KoopmanLoss(**OmegaConf.to_container(lcfg))
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
res = {
    "repo": args.repo,
    "n_params": sum(p.numel() for p in model.parameters()),
    "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
}


def train_step(inp, tgt, sync_timing=False):
    """Mirror of Trainer._run_one_epoch body (fp32, as configs/training/tra.yaml)."""
    inp, tgt = inp.to(dev), tgt.to(dev)
    opt.zero_grad(set_to_none=True)
    keys = list(model.data_variables.keys())
    noise = torch.randn_like(inp.select(*keys)) * 0.005
    noisy = inp.clone()
    for k in keys:
        noisy[k] = noisy[k] + noise[k]
    noisy = noisy.apply(
        lambda t: torch.nan_to_num(t, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)
    )
    out = model(noisy, tgt["seq_length"], drop_prob=0.0)
    x_true_recon = TensorDict(
        {k: inp[k][:, -1] for k in keys}, batch_size=inp.batch_size[0]
    )
    true_lat = Trainer.true_latent_encoding(None, tgt, model)
    loss = crit(
        koopman_operator=model.koopman_operator,
        x_recon=out.x_recon,
        x_preds=out.x_preds,
        latent_pred=out.z_preds,
        x_true=x_true_recon,
        x_future=tgt,
        true_latents=true_lat,
        reynolds=out.reynolds,
        disturbed_latents=out.disturbed_latents,
        dz_dt=out.dz_dt,
        dz_dt_disturbed=out.dz_dt_disturbed,
    )
    loss.total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    return loss


# ---------------- 1. gradient flow ----------------
model.train()
loss = train_step(*batches[0])
res["first_loss"] = {
    k: v
    for k, v in loss.metrics.items()
    if k
    in (
        "total_loss",
        "loss_recon",
        "loss_pred",
        "loss_latent",
        "loss_phys",
        "stability",
    )
}
groups: dict = defaultdict(lambda: [0, 0, 0])  # [n_tensors, n_none, n_allzero]
dead = []
for name, p in model.named_parameters():
    top = ".".join(name.split(".")[:2])
    groups[top][0] += 1
    if p.grad is None:
        groups[top][1] += 1
        dead.append((name, "None", p.numel()))
    elif float(p.grad.abs().max()) == 0.0:
        groups[top][2] += 1
        dead.append((name, "zero", p.numel()))
res["grad_groups[n,none,zero]"] = dict(groups)
res["params_without_gradient"] = dead
res["n_params_without_gradient"] = sum(d[2] for d in dead)
opt.step()

# ---------------- 2. checkpoint round trip ----------------
ck = Path(args.scratch) / "roundtrip.pth"
torch.save({"model_state_dict": model.state_dict()}, ck)
m2 = build()
m2.load_state_dict(
    torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"],
    strict=True,
)
model.eval()
m2.eval()
with torch.no_grad():
    i0, t0 = batches[1][0].to(dev), batches[1][1].to(dev)
    a = model(i0, t0["seq_length"])
    b = m2(i0, t0["seq_length"])
    res["roundtrip_max_abs_diff"] = max(
        float((a.x_preds[k] - b.x_preds[k]).abs().max()) for k in model.data_variables
    )
    a2 = model(i0, t0["seq_length"])
    res["eval_determinism_max_abs_diff"] = max(
        float((a.x_preds[k] - a2.x_preds[k]).abs().max()) for k in model.data_variables
    )
ck.unlink()
model.train()


# ---------------- 3. step timing A/B (interleaved) ----------------
def timed(n, anomaly, hooks):
    handles = []
    if hooks:  # the trainer's "NaN sniper": one .any() host sync per module output

        def nan_hook(module, inputs, outputs):
            if isinstance(outputs, torch.Tensor) and torch.isnan(outputs).any():
                raise RuntimeError("nan")

        handles = [m.register_forward_hook(nan_hook) for m in model.modules()]
    torch.autograd.set_detect_anomaly(anomaly)
    ts = []
    for s in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        train_step(*batches[s % 4])
        opt.step()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    torch.autograd.set_detect_anomaly(False)
    for h in handles:
        h.remove()
    ts = sorted(ts[1:])
    return ts[len(ts) // 2]


timed(3, False, False)  # warm-up
torch.cuda.reset_peak_memory_stats()
rows = defaultdict(list)
for rep in range(3):
    for name, an, hk in [
        ("as_trainer(anomaly+nan_hooks)", True, True),
        ("no_anomaly", False, True),
        ("no_anomaly_no_hooks", False, False),
    ]:
        rows[name].append(timed(max(4, args.steps // 3), an, hk))
res["median_step_seconds_by_config"] = {
    k: sorted(v)[len(v) // 2] for k, v in rows.items()
}
res["peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
res["n_nan_hook_modules"] = sum(1 for _ in model.modules())

# ---------------- 4. phase breakdown with the model's own timers -------------
model.timings.clear()
phase = defaultdict(list)
for s in range(6):
    train_step(*batches[s % 4])
    opt.step()
    for k, v in model.timings.items():
        phase[k].append(v)
res["forward_phase_ms_median(model.timings)"] = {
    k: sorted(v)[len(v) // 2] for k, v in phase.items()
}

# backward vs forward split
for s in range(2):
    train_step(*batches[s % 4])
fw, bw = [], []
for s in range(6):
    inp, tgt = batches[s % 4]
    torch.cuda.synchronize()
    t = time.perf_counter()
    inp, tgt = inp.to(dev), tgt.to(dev)
    out = model(inp, tgt["seq_length"])
    true_lat = Trainer.true_latent_encoding(None, tgt, model)
    keys = list(model.data_variables.keys())
    lres = crit(
        koopman_operator=model.koopman_operator,
        x_recon=out.x_recon,
        x_preds=out.x_preds,
        latent_pred=out.z_preds,
        x_true=TensorDict(
            {k: inp[k][:, -1] for k in keys}, batch_size=inp.batch_size[0]
        ),
        x_future=tgt,
        true_latents=true_lat,
        reynolds=out.reynolds,
        disturbed_latents=out.disturbed_latents,
        dz_dt=out.dz_dt,
        dz_dt_disturbed=out.dz_dt_disturbed,
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    lres.total_loss.backward()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    fw.append(t1 - t)
    bw.append(t2 - t1)
    opt.zero_grad(set_to_none=True)
res["forward_incl_loss_s_median"] = sorted(fw)[3]
res["backward_s_median"] = sorted(bw)[3]

print(json.dumps(res, indent=1, default=str))
if args.json:
    json.dump(res, open(args.json, "w"), indent=1, default=str)

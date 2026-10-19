"""Score a legacy KoopmanAutoencoder checkpoint with the unified TRA protocol (same cases,
alignment and score() as eval_ctkae.py / benchmark_suite.py).

    python eval_legacy_tra.py --repo KAE_DIR --ckpt CKPT.pth [--model_yaml MODEL.yaml] --json out.json

The model config comes from the checkpoint's run_metadata/config when present (audit
REPRO-02), otherwise from --model_yaml (e.g. the run's saved continous_linear_128_tra.yaml).
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
import yaml

ap = argparse.ArgumentParser()
ap.add_argument(
    "--repo", default="/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder"
)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--model_yaml", default=None)
ap.add_argument("--json", default=None)
args = ap.parse_args()
sys.path.insert(0, args.repo)
sys.path.insert(0, str(Path(__file__).parent))
warnings.simplefilter("ignore")

from tensordict import TensorDict  # noqa: E402
from benchmark_suite import load_gt, score, train_climatology  # noqa: E402
from models.autoencoder import KoopmanAutoencoder  # noqa: E402
from models.ct_data import VARS, load_eval_case, to_acdm_layout  # noqa: E402
from models.networks import TransformerConfig  # noqa: E402

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = (ck.get("run_metadata") or {}).get("config") or ck.get("config")
m = cfg["model"] if cfg else yaml.safe_load(open(args.model_yaml))
input_frames = cfg["data"]["input_sequence_length"] if cfg else 2
kw = dict(m.get("conv_kwargs", {}))
model = KoopmanAutoencoder(
    data_variables={v: 1 for v in VARS},
    input_frames=input_frames,
    height=m["height"],
    width=m["width"],
    latent_dim=m["latent_dim"],
    cond_embedding_dim=m["cond_embedding_dim"],
    cond_type=m["cond_type"],
    operator_mode=m["operator_mode"],
    hidden_dims=m["hidden_dims"],
    transformer_config=TransformerConfig(**m["transformer"]),
    predict_cond=m["predict_cond"],
    cond_grad_enabled=m.get("cond_grad_enabled", False),
    disturb_std=None,
    is_continuous=m["is_continuous"],
    operator_type=m.get("operator_type", "continuous"),
    rank=m["rank"],
    spectral=m["spectral"],
    cond_expansion_type="Ma",
    use_attention=m["use_attention"],
    honor_conv_kwargs=m.get("honor_conv_kwargs", False),
    dt=m.get("dt", 0.1),
    **kw,
)
model.load_state_dict(ck["model_state_dict"], strict=True)
dev = torch.device("cuda")
model = model.to(dev).eval()
data = Path(args.repo) / "data/acdm/128_tra"
clim = train_climatology()
out = {"ckpt": args.ckpt}
with torch.no_grad():
    for case in ["interp", "extrap", "longer"]:
        x, ma, mask = load_eval_case(str(data / f"gt_{case}.nc"), dev)
        S, T = x.shape[:2]
        preds = []
        for s in range(S):
            td = TensorDict(
                {v: x[s : s + 1, 2 - input_frames : 2, i] for i, v in enumerate(VARS)},
                batch_size=[1, input_frames],
            )
            td["obstacle_mask"] = mask.expand(1, input_frames, -1, -1)
            td["cond_input"] = ma[s : s + 1, None].expand(1, input_frames)
            o = model(td, T - 2)
            preds.append(torch.stack([o.x_preds[v][0] for v in VARS], 1))
        full = torch.cat([x[:, :2], torch.stack(preds)], 1)
        gt, fluid = load_gt(case)
        r = score(to_acdm_layout(full, ma)[None, :, :, :4], gt, fluid, clim)
        out[case] = r
        print(
            f"{case:7s} mse {r['mse_pred']:.3f}  acc<0.6@{r['first_acc_below_0.6']}  E {r['energy_ratio_at']}"
        )
if args.json:
    Path(args.json).write_text(json.dumps(out))

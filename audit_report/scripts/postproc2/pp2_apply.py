"""Apply post-processing settings (fixed on the proxy) to a FULL checkpoint on the test cases.

Saves test forecasts (normalised, fp16) per case for each requested variant:
    VARIANT = z0:sigma[:mirror]   z0 = enc | ref<lr>_<steps>;  sigma = float | cal (RUN/mix_sigma.json)
e.g.  enc:cal   enc:cal:mirror   ref0.05_25:0.06:mirror

    python pp2_apply.py RUN_DIR OUT.npz VARIANT [VARIANT ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from pp2_lab import REPO, dev, forecast, mirror, rec_loss  # noqa: E402

from eval_ctkae import load_model  # noqa: E402
from models.ct_data import load_eval_case  # noqa: E402


def refine(m, z0, x2, mask, ma, lr, steps):
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        rec_loss(m, z, x2, mask, ma).sum().backward()
        opt.step()
    return z.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out")
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--mem_gb", type=float, default=3.0)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(
        a.mem_gb / (torch.cuda.get_device_properties(0).total_memory / 2**30)
    )
    run = Path(a.run_dir)
    m, _ = load_model(run, "last.pth", dev)
    for p in m.parameters():
        p.requires_grad_(False)
    sig_cal = json.load(open(run / "mix_sigma.json"))["sigma"]
    out = {}
    for case in ["extrap", "interp", "longer"]:
        x, ma, mask = load_eval_case(str(REPO / f"data/acdm/128_tra/gt_{case}.nc"), dev)
        x2 = x[:, :2]
        K = x.shape[1] - 2
        taus = 0.1 * torch.arange(1, K + 1, device=dev, dtype=torch.float32)
        with torch.no_grad():
            z_enc = m.encode(x2, mask, ma)
            mmask = mask.flip(-2)
            z_mir = m.encode(mirror(x2), mmask, ma)
        for v in a.variants:
            parts = v.split(":")
            zk, sg = parts[0], (sig_cal if parts[1] == "cal" else float(parts[1]))
            use_mirror = len(parts) > 2 and parts[2] == "mirror"
            if zk == "enc":
                z, zm = z_enc, z_mir
            else:
                lr, steps = zk[3:].split("_")
                z = refine(m, z_enc, x2, mask, ma, float(lr), int(steps))
                zm = refine(m, z_mir, mirror(x2), mmask, ma, float(lr), int(steps))
            fields = []
            for i in range(x.shape[0]):
                sl = slice(i, i + 1)
                p = forecast(m, z[sl], taus[None], ma[sl], mask, sg)
                if use_mirror:
                    pm = forecast(m, zm[sl], taus[None], ma[sl], mmask, sg)
                    p = 0.5 * (p + mirror(pm) * mask)
                fields.append(p.cpu().numpy().astype(np.float16))
            out[f"{case}|{v}"] = np.concatenate(fields)
            print(case, v, "sigma", sg, flush=True)
    np.savez(a.out, **out)
    print("saved", a.out)


if __name__ == "__main__":
    main()

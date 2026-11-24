# mypy: disable-error-code="arg-type"
"""Roll every model forward from a PERTURBED initial condition.

Two questions at once:

1. **Correctness.**  A trained surrogate handed a slightly wrong state should produce a
   slightly wrong flow, not garbage and not a blow-up.  If a 1% perturbation destroys the
   rollout, the model (or the way it is being driven) is wrong.

2. **Why a 4D-Var analysis can be noise.**  The U-Net and FNO analyses at $t_0$ come out as
   speckle rather than flow.  That is only possible if the dynamics DESTROY such states
   quickly -- then the observations, which all lie downstream of $t_0$, cannot see them and
   the cost does not penalise them.  Perturbing with white noise and watching whether the
   error grows or decays measures exactly that.

Perturbations are scaled relative to the per-channel standard deviation of the field, and
applied only outside the obstacle.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from data_assimilation.tra.bridge import REGIMES, PhysicalData, REPO, rel_l2
from data_assimilation.tra.adapters import TurbpredAdapter
from data_assimilation.tra.kae_adapter import KAEAdapter
from data_assimilation.tra.verify_adapters import MODELS, DIFF_OPTS

ORDER = ["KAE", "UNet", "FNO", "ACDM", "ACDM-ncn"]


def make_pert(x, kind, amp, mask, gen):
    """Perturbation of relative size `amp`, either white or smooth (low-wavenumber)."""
    sd = x.reshape(x.shape[0], x.shape[1], -1).std(-1)[..., None, None]
    n = torch.randn(x.shape, device=x.device, generator=gen)
    if kind == "smooth":
        # keep only the lowest wavenumbers -> a perturbation the flow can carry
        F = torch.fft.rfft2(n)
        H, W = F.shape[-2], F.shape[-1]
        keep = torch.zeros_like(F.real)
        keep[..., : max(1, H // 16), : max(1, W // 16)] = 1.0
        n = torch.fft.irfft2(F * keep, s=x.shape[-2:])
        n = n / n.reshape(n.shape[0], n.shape[1], -1).std(-1)[..., None, None]
    p = amp * sd * n
    if mask is not None:
        p = p * mask[:, None]
    return x + p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--channel", type=int, default=3)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--amps", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.20])
    ap.add_argument("--kind", default="white", choices=["white", "smooth"])
    ap.add_argument(
        "--perturb-frames",
        default="window",
        choices=["window", "t0"],
        help=(
            "'window' perturbs EVERY frame of the conditioning window, "
            "which is the only like-for-like choice: U-Net and FNO condition "
            "on one frame (t_0) while the KAE and both samplers condition on "
            "two, so perturbing t_0 alone leaves the k=2 models with half "
            "their input CLEAN and hands them an easier problem. 't0' "
            "reproduces that earlier, biased behaviour for comparison."
        ),
    )
    ap.add_argument(
        "--where",
        default="state",
        choices=["state", "control"],
        help=(
            "'state' perturbs the physical field, which the KAE then "
            "ENCODES -- its 128-d latent discards most white noise before "
            "any dynamics run, so KAE looks almost untouched. That measures "
            "encoder filtering, not dynamical damping. 'control' instead "
            "perturbs each method's OWN control variable by the same "
            "RELATIVE amount (the KAE's latent, the others' input frame), "
            "which is the like-for-like test."
        ),
    )
    ap.add_argument(
        "--kae-run",
        default=("model_outputs_tra/continous_linear_128/" "run-20260821_031121"),
    )
    a = ap.parse_args()

    dev = torch.device("cuda")
    reg = REGIMES["tra"]
    d = PhysicalData("data/acdm/128_tra/gt_interp.nc", "tra", dev)
    sim = torch.arange(a.batch)
    t0 = torch.full((a.batch,), 4, dtype=torch.long)
    par, om = d.params_for(sim), d.mask_for(sim)
    T = a.frames
    truth = torch.stack([d.frames(sim, t0 + k) for k in range(T)], dim=1)
    base = REPO / "autoreg_pde_diffusion" / "pretrained_models" / "models_tra"
    gen = torch.Generator(device=dev).manual_seed(0)

    out, store = {}, {
        "amps": np.array(a.amps),
        "kind": np.array(a.kind),
        "where": np.array(a.where),
        "perturb_frames": np.array(a.perturb_frames),
    }

    # The perturbed window is drawn ONCE, before the model loop, and every model is handed
    # the same one. It used to be drawn inside the loop from a shared generator, so each
    # model saw a DIFFERENT noise realisation -- harmless for the conclusion (white noise
    # of a given amplitude has a near-deterministic norm) but a violation of the campaign's
    # one rule, that every method sees the identical problem.
    k_max = max(4, 1)
    windows = {}
    for amp in a.amps:
        w = d.window(sim, t0 - (k_max - 1), k_max).clone()
        if amp > 0:
            # Every frame of the window is perturbed independently, at the same RELATIVE
            # amplitude. Perturbing only w[:, -1] (t_0) is not like-for-like: U-Net and FNO
            # condition on a single frame, so t_0 IS their whole input, while the KAE and
            # both samplers condition on two and would keep a clean frame -- half their
            # input, handed to them for free, on top of the KAE's encoder already
            # projecting most of the noise out.
            n_pert = w.shape[1] if a.perturb_frames == "window" else 1
            for i in range(w.shape[1] - n_pert, w.shape[1]):
                w[:, i] = make_pert(w[:, i], a.kind, amp, om, gen)
        windows[amp] = w
        # what every model is actually handed at t_0, stored so a figure can show it
        store[f"input__amp{amp:g}"] = w[:, -1].cpu().numpy()
        store[f"input_window__amp{amp:g}"] = w.cpu().numpy()

    print(
        f"perturbation: {a.kind} noise on the {a.where}, "
        f"{'EVERY frame of the conditioning window' if a.perturb_frames == 'window' else 'the t_0 frame only'}\n"
    )
    hdr = f"{'model':9s} " + " ".join(f"{f'amp={x:g}':>12s}" for x in a.amps)
    print(hdr + "   (rel-L2 at the FINAL frame)")
    for m in ORDER:
        if m == "KAE":
            ad = KAEAdapter(a.kae_run, reg, dev)
        else:
            ad = TurbpredAdapter(
                base / MODELS[m] / "Model.pth", reg, dev, DIFF_OPTS.get(m), name=m
            )
        k0 = ad.n_control_frames
        row, curves = [], []
        for amp in a.amps:
            # the same drawn window for every model, trimmed to this model's own length
            w = windows[amp][:, k_max - k0 :].clone()
            if a.where == "control" and m == "KAE" and amp > 0:
                # the KAE's control is the LATENT, so the physical field must go in clean
                # and the perturbation is applied after encoding, below
                w = d.window(sim, t0 - (k0 - 1), k0).clone()
            with torch.no_grad():
                if m == "KAE":
                    z = ad.encode(w, par)
                    if amp > 0 and a.where == "control":
                        # same RELATIVE size, applied to the KAE's OWN control variable:
                        # perturbing the physical field instead lets the encoder filter it
                        # out before any dynamics run, which is not a like-for-like test
                        z = z + amp * z.std() * torch.randn(
                            z.shape, device=z.device, generator=gen
                        )
                    K = ad.generator(par)
                    roll = torch.stack(
                        [
                            ad.decode(ad.propagate(z, k * ad.dt_train, K=K))
                            for k in range(T)
                        ],
                        dim=1,
                    )
                else:
                    tr = ad.rollout(ad.to_model(w), T, par, checkpoint_every=0)
                    roll = tr[:, k0 - 1 : k0 - 1 + T]
            e = (
                torch.stack([rel_l2(roll[:, k], truth[:, k], om) for k in range(T)])
                .mean(1)
                .cpu()
                .numpy()
            )
            curves.append(e)
            row.append(e[-1])
            store[f"{m}__amp{amp:g}__roll"] = roll.cpu().numpy()
        store[f"{m}__curves"] = np.stack(curves)
        out[m] = row
        print(f"{m:9s} " + " ".join(f"{v:12.4f}" for v in row))
        del ad
        torch.cuda.empty_cache()

    store["truth"] = truth.cpu().numpy()
    if om is not None:
        store["mask"] = om.cpu().numpy()
    tag = "" if a.perturb_frames == "window" else "_t0only"
    np.savez_compressed(
        f"da_results_tra/perturbed_{a.kind}_{a.where}{tag}.npz", **store
    )

    # how the perturbation itself evolves: error at frame k relative to error at frame 0
    print(
        f"\ngrowth of the perturbation (error at frame k / error at frame 0), "
        f"amp={a.amps[-1]:g}:"
    )
    print(
        f"{'model':9s} "
        + " ".join(f"{'fr'+str(k):>8s}" for k in (0, 1, 2, 5, 12, T - 1))
    )
    inj = max(
        float((store[f"{m}__curves"][-1] - store[f"{m}__curves"][0])[0]) for m in ORDER
    )
    for m in ORDER:
        c = store[f"{m}__curves"][-1]
        c0 = store[f"{m}__curves"][0]  # unperturbed baseline
        g = c - c0
        # A model whose state never received the perturbation has g[0] at the noise floor.
        # Dividing by that is meaningless -- with the clamp at 1e-12 the KAE printed a
        # growth of 1.0e9 -- so say what actually happened instead of printing a number.
        if g[0] < 0.1 * inj:
            print(
                f"{m:9s}    no measurable perturbation at frame 0 "
                f"(excess {g[0]:+.4f} against {inj:.4f} injected)"
            )
            continue
        gg = g.clip(min=1e-12)
        print(
            f"{m:9s} "
            + " ".join(f"{gg[k] / gg[0]:8.3f}" for k in (0, 1, 2, 5, 12, T - 1))
        )
    print(f"\nsaved -> da_results_tra/perturbed_{a.kind}_{a.where}{tag}.npz")


if __name__ == "__main__":
    main()

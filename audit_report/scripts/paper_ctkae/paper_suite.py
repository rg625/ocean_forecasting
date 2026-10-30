"""The paper's transonic experiments, re-run at inference for the two CT-KAE candidates.

Models: the reference and back001, each as a single model (seed 0, single decode) and with
the adopted post-processing ("+pp": 3-seed average of the mixture readout with mirror
averaging, plus climatology shrinkage on the 60/240-step cases; no shrinkage on the
1000-step rollout, whose gains were only fitted to lead 238).

    python paper_suite.py fields     # ACDM-layout test forecasts -> results/paper_ctkae/fields
    python paper_suite.py long1000   # 1000-step rollout: v_x correlation and rel. L2 per frame
    python paper_suite.py spectral   # band errors of the |v| spectrum (spectral_analysis.py)
    python paper_suite.py solvers    # latent ODE integrated numerically vs e^{tau A}
    python paper_suite.py eigen      # generator eigenvalues across Mach
    python paper_suite.py sharp | spectral_sharp | long1000_sharp
                                     # round-3 '+sharp' readout (postproc3/sharp.py)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
IMPL = HERE.parents[2]
KAE = IMPL / "koopman_autoencoder"
SCR = IMPL / "audit_report/scripts"
OUT = IMPL / "audit_report/results/paper_ctkae"
PP1 = Path(
    "/tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/093163e5-e75d-47aa-b04b-9bb0dda37f82/scratchpad/res/pp"
)
PP2 = Path(
    "/tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/093163e5-e75d-47aa-b04b-9bb0dda37f82/scratchpad/res/pp2"
)
AUTOREG = Path("/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion")
for p in (KAE, SCR, SCR / "postproc2", SCR / "postproc3", PP1):
    sys.path.insert(0, str(p))

from eval_ctkae import load_model  # noqa: E402
from models.ct_data import load_eval_case, to_acdm_layout  # noqa: E402
from models.ct_kae import mixture_nodes  # noqa: E402

dev = torch.device("cuda")
CASES = ["extrap", "interp", "longer"]
MODELS = ["ref", "back001"]
LABEL = {"ref": "CT-KAE ref", "back001": "CT-KAE back001"}


def run_dir(key):
    r = Path(
        open(KAE / f"model_outputs_ctkae/ablation_state/{key}.done").read().strip()
    )
    return r if r.is_absolute() else KAE / r


def save_acdm(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, arr_0=arr[None, None].astype(np.float16))


# ----------------------------------------------------------------------------- fields
def fields():
    import pp_analyze as P1

    for t in MODELS:
        cks = [P1.Ckpt(f"{t}_s{s}") for s in range(3)]
        aps = [np.load(PP2 / f"apply_{t}_s{s}.npz") for s in range(3)]
        for case in CASES:
            xin, ma, _, _ = P1.gt_case(case)
            single = cks[0].d[f"{case}_pred"][0].astype(np.float32)  # sigma = 0, seed 0
            clim = cks[0].d[f"{case}_clim"][:, None]
            pp = 0.0
            for ck, ap in zip(cks, aps):
                x = ap[f"{case}|enc:cal:mirror"].astype(np.float32)
                g = ck.gains("clim", np.full(238, ck.icar), per_band=False)[..., 0]
                g = g[: x.shape[1]][None, :, :, None, None]
                mask = (np.abs(x).sum((0, 1, 2)) > 0).astype(np.float32)
                pp = pp + (clim + g * (x - clim)) * mask / 3
            for name, x in [(t, single), (f"{t}+pp", pp)]:
                full = torch.cat(
                    [xin.float(), torch.from_numpy(np.ascontiguousarray(x))], 1
                )
                save_acdm(
                    OUT / "fields" / case / f"{name}.npz", to_acdm_layout(full, ma)
                )
                print(case, name, "mse_all", round(P1.mse_all(case, x), 3), flush=True)


# ------------------------------------------------------------------- 1000-step rollout
def _mirror(x):
    y = x.flip(-2).clone()
    y[..., 1, :, :] *= -1
    return y


@torch.no_grad()
def _decode(m, z, ma, mask, chunk=64):
    N, K, d = z.shape
    flat, mk = z.reshape(-1, d), ma.repeat_interleave(K)
    out = []
    for i in range(0, len(flat), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(m.decode(flat[i : i + chunk], mask, mk[i : i + chunk]).float())
    return torch.cat(out).view(N, K, *out[0].shape[1:]) * mask


@torch.no_grad()
def _forecast(m, x2, ma, mask, taus, sigma, mirror):
    def one(xw, mk):
        z0 = m.encode(xw, mk, ma)
        acc = 0.0
        for sq, wq in mixture_nodes(sigma):
            acc = acc + wq * _decode(m, m.forecast(z0, taus, ma, sq), ma, mk)
        return acc

    p = one(x2, mask)
    if mirror:
        p = 0.5 * (p + _mirror(one(_mirror(x2), mask.flip(-2))) * mask)
    return p


def _corr_l2(pred, gt, fluid):
    """per sequence and frame: Pearson correlation and relative L2 of v_x over fluid cells.
    pred, gt [S, T, F, 128, 64] physical (ACDM layout, field 0 = v_x); fluid [S, 128, 64]
    """
    f = fluid[:, None].astype(bool)
    p, g = pred[:, :, 0], gt[:, :, 0]
    w = np.broadcast_to(f, p.shape)
    pm = np.where(w, p, 0).sum((-1, -2)) / w.sum((-1, -2))
    gm = np.where(w, g, 0).sum((-1, -2)) / w.sum((-1, -2))
    pa, ga = np.where(w, p - pm[..., None, None], 0), np.where(
        w, g - gm[..., None, None], 0
    )
    corr = (pa * ga).sum((-1, -2)) / np.sqrt(
        (pa**2).sum((-1, -2)) * (ga**2).sum((-1, -2))
    )
    l2 = np.sqrt(
        (np.where(w, p - g, 0) ** 2).sum((-1, -2))
        / (np.where(w, g, 0) ** 2).sum((-1, -2))
    )
    return corr, l2


def long1000():
    gtd = torch.load(
        AUTOREG / "results/longer/groundTruth.dict",
        map_location="cpu",
        weights_only=False,
    )
    gt = gtd["data"].numpy()  # [S, 998, 5, 128, 64]
    fluid = gtd["obsMask"].numpy()  # 1 = fluid
    x, ma, mask = load_eval_case(str(KAE / "data/acdm/128_tra/gt_longer_1000.nc"), dev)
    S, T = gt.shape[:2]
    # alignment check: the .nc frames are the ground-truth frames
    phys = to_acdm_layout(x[:, :T].cpu(), ma.cpu())
    fl = fluid[:, None, None].astype(bool)
    assert (
        np.abs(np.where(fl, phys[:, :5, :4] - gt[:, :5, :4], 0)).max() < 1e-3
    ), "misaligned"
    # input: frames 0 and 2 (0.1 s apart; the file is 0.05 s per frame, audit DATA-03);
    # frame k >= 3 is the forecast at tau = 0.05 (k - 2)
    taus = 0.05 * torch.arange(1, T - 2, device=dev, dtype=torch.float32)[None].expand(
        S, -1
    )
    x2 = x[:, [0, 2]]
    res, snaps = {}, {}
    steps = [99, 249, 749, 996]
    snaps["truth"] = gt[0, steps, 3]
    for t in MODELS:
        ms = [load_model(run_dir(f"{t}_s{s}"), "last.pth", dev)[0] for s in range(3)]
        sig = [
            json.load(open(run_dir(f"{t}_s{s}") / "mix_sigma.json"))["sigma"]
            for s in range(3)
        ]
        for name, members, sg, mir in [
            (t, ms[:1], [0.0], False),
            (f"{t}+pp", ms, sig, True),
        ]:
            p = sum(
                _forecast(m, x2, ma, mask, taus, s_, mir) for m, s_ in zip(members, sg)
            ) / len(members)
            full = torch.cat([x[:, :3], p], 1)[:, :T].cpu()
            pa = to_acdm_layout(full, ma.cpu())
            c, l2 = _corr_l2(pa, gt, fluid)
            res[name] = {"corr": c.tolist(), "l2": l2.tolist()}
            snaps[name] = pa[0, steps, 3]
            print(
                "long1000",
                name,
                "corr@100/250/997",
                [round(float(c[:, i].mean()), 3) for i in (99, 249, 996)],
                flush=True,
            )
            del p, full, pa
    for name, f in [
        ("ACDM", "acdm-r20.npz"),
        ("Original KAE", "continous_linear_128_tra.npz"),
    ]:
        a = np.load(AUTOREG / "results/longer" / f)["arr_0"]
        a = a.reshape(-1, *a.shape[2:])  # [M*E, S, T, ...]
        cs, ls = zip(
            *[
                _corr_l2(a[i, :, :T].astype(np.float32), gt, fluid)
                for i in range(a.shape[0])
            ]
        )
        res[name] = {"corr": np.mean(cs, 0).tolist(), "l2": np.mean(ls, 0).tolist()}
        snaps[name] = a[0, 0, steps, 3].astype(np.float32)
        print(
            "long1000",
            name,
            "corr@100/250/997",
            [round(float(np.mean(cs, 0)[:, i].mean()), 3) for i in (99, 249, 996)],
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"frame_dt": 0.05, "first_forecast_frame": 3, "models": res},
        open(OUT / "long1000.json", "w"),
    )
    np.savez_compressed(
        OUT / "long1000_pressure_snapshots.npz", steps=np.array(steps), **snaps
    )


# ------------------------------------------------------------------------------ spectra
def spectral():
    sys.path.insert(0, str(AUTOREG / "src"))
    import spectral_analysis as SA

    base = {
        "ACDM": "acdm-r20.npz",
        "U-Net-ut": "unet-m8.npz",
        "FNO-16": "fno-16modes-m2.npz",
    }
    out = {}
    for case, wins in [
        ("longer", [(0, 240), (0, 60), (60, 120), (120, 180), (180, 240)]),
        ("interp", [(0, 60)]),
        ("extrap", [(0, 60)]),
    ]:
        mods = dict(base)
        for t in MODELS:
            for name in (t, f"{t}+pp"):
                mods[name] = str(OUT / "fields" / case / f"{name}.npz")
        rows, curves = SA.analyse(case, mods, wins)
        out[case] = {"rows": rows, "curves": curves}
    json.dump(out, open(OUT / "spectral.json", "w"), indent=0)
    for case, d in out.items():
        print(f"== {case}")
        for r in d["rows"]:
            if not r.get("diverged"):
                print(
                    f"  {r['model']:16s} {r['window']:8s} low {r['err_low']:.3f} mid {r['err_mid']:.3f} high {r['err_high']:.3f} slope {r['slope_pred']:.2f} (gt {r['slope_gt']:.2f})"
                )


# ------------------------------------------------------- round-3 sharpened mirror average
PP3 = Path(
    "/tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/cfcfc896-3b58-4c33-b471-e1efcfb360e4/scratchpad/pp3px"
)


def sharp_fields():
    """'+sharp': 3-seed average of plain + mirrored single decodes, re-sharpened to the mean
    single-decode band power (postproc3/sharp.py) -> fields/<case>/<t>+sharp.npz, MSE, LSiM
    """
    import pp_analyze as P1
    import sharp as SH
    from lsim_score import lsim

    rb = SH.radial_bins(64, 128, torch.device("cpu"))
    res: dict = {}
    for t in MODELS:
        cks = [P1.Ckpt(f"{t}_s{s}") for s in range(3)]
        for case in CASES:
            xin, ma, _, _ = P1.gt_case(case)
            M = torch.from_numpy(cks[0].d[f"{case}_clim"][:, None].astype(np.float32))
            avg, e = 0.0, 0.0
            for s, ck in enumerate(cks):
                p = torch.from_numpy(ck.d[f"{case}_pred"][0].astype(np.float32))
                mir = np.load(PP3 / f"apply0_{t}_s{s}.npz")[f"{case}|enc:0:mirror"]
                mir = torch.from_numpy(mir.astype(np.float32))
                mask = (p.abs().sum((0, 1, 2)) > 0).float()
                pm = (2 * mir - p) * mask  # mirrored single decode
                e = e + (SH.band_power(p, M, rb) + SH.band_power(pm, M, rb)) / 6
                avg = avg + mir / 3
            x = SH.sharpen(avg, e, M, mask, rb).numpy()
            full = torch.cat([xin.float(), torch.from_numpy(x)], 1)
            pa = to_acdm_layout(full, ma)
            save_acdm(OUT / "fields" / case / f"{t}+sharp.npz", pa)
            r = {
                "mse_all": P1.mse_all(case, x),
                "lsim": lsim(pa[None, :, :, :4], case)[0],
            }
            res.setdefault(f"{t}+sharp", {})[case] = r
            print(case, t, "+sharp", r, flush=True)
    json.dump(res, open(OUT / "sharp_fields.json", "w"), indent=1)


def spectral_sharp():
    sys.path.insert(0, str(AUTOREG / "src"))
    import spectral_analysis as SA

    out = {}
    for case, wins in [
        ("longer", [(0, 240), (0, 60), (60, 120), (120, 180), (180, 240)]),
        ("interp", [(0, 60)]),
        ("extrap", [(0, 60)]),
    ]:
        mods = {
            f"{t}+sharp": str(OUT / "fields" / case / f"{t}+sharp.npz") for t in MODELS
        }
        rows, curves = SA.analyse(case, mods, wins)
        out[case] = {"rows": rows, "curves": curves}
    json.dump(out, open(OUT / "spectral_sharp.json", "w"), indent=0)
    for case, d in out.items():
        for r in d["rows"]:
            print(
                f"{case:7s} {r['model']:16s} {r['window']:8s} low {r['err_low']:.3f} mid {r['err_mid']:.3f} high {r['err_high']:.3f} slope {r['slope_pred']:.2f}"
            )


def long1000_sharp():
    """'+sharp' on the 1000-step rollout (as long1000; climatology of the gt_longer sim with
    the same Mach number)"""
    import pp_analyze as P1
    import sharp as SH

    gtd = torch.load(
        AUTOREG / "results/longer/groundTruth.dict",
        map_location="cpu",
        weights_only=False,
    )
    gt, fluid = gtd["data"].numpy(), gtd["obsMask"].numpy()
    x, ma, mask = load_eval_case(str(KAE / "data/acdm/128_tra/gt_longer_1000.nc"), dev)
    ma_l = P1.gt_case("longer")[1].cpu()
    idx = [
        int((ma_l - v).abs().argmin()) for v in ma.cpu()
    ]  # sims of gt_longer by Mach
    assert torch.allclose(ma_l[idx], ma.cpu()), "no matching climatology"
    S, T = gt.shape[:2]
    taus = 0.05 * torch.arange(1, T - 2, device=dev, dtype=torch.float32)[None].expand(
        S, -1
    )
    x2 = x[:, [0, 2]]
    rb = SH.radial_bins(64, 128, dev)
    steps = [99, 249, 749, 996]
    res, snaps = {}, {}
    for t in MODELS:
        M = P1.Ckpt(f"{t}_s0").d["longer_clim"][idx][:, None]
        M = torch.from_numpy(M).float().to(dev)
        avg, e = 0.0, 0.0
        for s in range(3):
            m = load_model(run_dir(f"{t}_s{s}"), "last.pth", dev)[0]
            p = _forecast(m, x2, ma, mask, taus, 0.0, False)
            pm = _mirror(_forecast(m, _mirror(x2), ma, mask.flip(-2), taus, 0.0, False))
            pm = pm * mask
            e = e + (SH.band_power(p, M, rb) + SH.band_power(pm, M, rb)) / 6
            avg = avg + (p + pm) / 6
            del m, p, pm
        out = SH.sharpen(avg, e, M, mask, rb)
        full = torch.cat([x[:, :3], out], 1)[:, :T].cpu()
        pa = to_acdm_layout(full, ma.cpu())
        c, l2 = _corr_l2(pa, gt, fluid)
        res[f"{t}+sharp"] = {"corr": c.tolist(), "l2": l2.tolist()}
        snaps[f"{t}+sharp"] = pa[0, steps, 3]
        print(
            "long1000",
            t,
            "+sharp corr@100/250/997",
            [round(float(c[:, i].mean()), 3) for i in (99, 249, 996)],
            flush=True,
        )
        del avg, e, out, full, pa
    json.dump(
        {"frame_dt": 0.05, "first_forecast_frame": 3, "models": res},
        open(OUT / "long1000_sharp.json", "w"),
    )
    np.savez_compressed(
        OUT / "long1000_sharp_pressure_snapshots.npz", steps=np.array(steps), **snaps
    )


# ------------------------------------------------------------------------------ solvers
def solvers():
    from torchdiffeq import odeint

    import pp_analyze as P1

    res: dict = {}
    grid = [0.05, 0.1, 0.15, 0.3, 0.5, 1.0]
    for t in MODELS:
        m, _ = load_model(run_dir(f"{t}_s0"), "last.pth", dev)
        for case in CASES:
            x, ma, mask = load_eval_case(
                str(KAE / f"data/acdm/128_tra/gt_{case}.nc"), dev
            )
            K = x.shape[1] - 2
            with torch.no_grad():
                z0 = m.encode(x[:, :2], mask, ma)
                e = m.embed_cond(ma)
                mu = m.fixed_point(e, z0.shape[0])
                A = m.generator_matrix(ma).float()  # [S, d, d], dy/dt = A y, y = z - mu
                y0 = (z0 - mu).float()
                tt = torch.cat(
                    [
                        torch.zeros(1, device=dev),
                        0.1 * torch.arange(1, K + 1, device=dev),
                    ]
                )

                def f(_t, y, A=A):
                    return torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)

                runs = (
                    [("exact", None, None)]
                    + [
                        (meth, h, None)
                        for meth in ("euler", "midpoint", "rk4")
                        for h in grid
                    ]
                    + [("dopri5", None, 1e-7)]
                )
                for meth, h, tol in runs:
                    if meth == "exact":
                        zs = m.propagate(z0, tt[1:][None].expand(len(z0), -1), ma)
                    else:
                        kw: dict = (
                            {"options": {"step_size": h}}
                            if h
                            else {"rtol": tol, "atol": tol}
                        )
                        ys = odeint(f, y0, tt, method=meth, **kw)[1:].transpose(
                            0, 1
                        )  # [S, K, d]
                        zs = mu[:, None] + ys
                    ok = bool(torch.isfinite(zs).all()) and float(zs.abs().max()) < 1e6
                    if ok:
                        p = _decode(m, zs, ma, mask)
                        v = P1.mse_all(case, p.cpu().numpy())
                    else:
                        v = float("inf")
                    key = f"{meth}" + (f"@{h:g}" if h else "")
                    res.setdefault(t, {}).setdefault(key, {})[case] = v
                    print(t, case, key, round(v, 3), flush=True)
    json.dump(res, open(OUT / "solvers.json", "w"), indent=1)


# ------------------------------------------------------------------------------ eigen
def eigen():
    res = {}
    mas = np.round(np.arange(0.45, 0.951, 0.01), 3)
    for t in MODELS:
        m, _ = load_model(run_dir(f"{t}_s0"), "last.pth", dev)
        with torch.no_grad():
            c = torch.tensor(mas, device=dev, dtype=torch.float32)
            g, w = m.gen.params(m.embed_cond(c), len(mas), m.norm_cond(c))
        res[t] = {
            "ma": mas.tolist(),
            "gamma": g.cpu().tolist(),
            "omega": w.cpu().tolist(),
        }
        print(t, "max gamma", float(g.max()), "max |omega|", float(w.abs().max()))
    json.dump(res, open(OUT / "eigen.json", "w"))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {
        "fields": fields,
        "long1000": long1000,
        "spectral": spectral,
        "sharp": sharp_fields,
        "spectral_sharp": spectral_sharp,
        "long1000_sharp": long1000_sharp,
        "solvers": solvers,
        "eigen": eigen,
    }[sys.argv[1]]()

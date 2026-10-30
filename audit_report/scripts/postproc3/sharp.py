"""Spectral re-sharpening of an averaged forecast (post-processing round 3).

The anomaly of the average about the climatology M is rescaled per (frame, channel, radial
wavenumber) so that its band power matches the mean band power of the single decodes that were
averaged; the phases of the average are kept and the gain is clipped to [1, GMAX]. Selected on
the proxies' held-out Mach numbers with the average = plain + mirrored single decode (pp3_proxy.py,
sharp(mirror,a1,c5)); the 3-seed version averages and matches over all 6 single decodes.
"""

import torch

GMAX = 5.0


def radial_bins(H, W, device):
    ky = torch.fft.fftfreq(H, d=1.0 / H, device=device)
    kx = torch.arange(W // 2 + 1, device=device, dtype=torch.float32)
    return torch.sqrt(ky[:, None] ** 2 + kx[None] ** 2).round().long()  # [H, Wr]


def band_power(x, M, rb):
    """anomaly band power [..., NB] of x [..., H, W]"""
    p = torch.fft.rfft2((x - M).double()).abs() ** 2
    nb = int(rb.max()) + 1
    out = torch.zeros(*p.shape[:-2], nb, dtype=p.dtype, device=p.device)
    return out.index_add_(-1, rb.flatten(), p.flatten(-2))


def sharpen(avg, e_single, M, mask, rb, gmax=GMAX):
    """avg [..., C, H, W]; e_single = mean single-decode band power, same leading shape"""
    F = torch.fft.rfft2((avg - M).double())
    nb = e_single.shape[-1]
    e_avg = torch.zeros_like(e_single).index_add_(
        -1, rb.flatten(), (F.abs() ** 2).flatten(-2)
    )[..., :nb]
    g = torch.clamp(torch.sqrt(e_single / e_avg.clamp_min(1e-30)), 1.0, gmax)
    out = M + torch.fft.irfft2(F * g[..., rb], s=avg.shape[-2:]).to(avg.dtype)
    return out * mask


def lpm(avg, singles, mask, ps=16, st=8, chunk=512):
    """Localized probability-matched mean (Clark 2017; Snook et al. 2020; post-processing round
    4, pp4_proxy.lpm): in 16x16 patches (stride 8, Hann-weighted overlap) the fluid values of avg
    are replaced, rank for rank, by the pooled values of the single decodes. avg and each single
    [..., H, W]; mask [H, W] fluid indicator; patches with fewer than 8 fluid cells are skipped.
    """
    import torch.nn.functional as F

    H, W = avg.shape[-2:]
    n = len(singles)
    fl = (mask > 0.5).to(avg.dtype)
    m_p = F.unfold(fl.reshape(1, 1, H, W), ps, stride=st) > 0.5  # [1, P, L]
    valid = (m_p.sum(1, keepdim=True) >= 8).to(avg.dtype)
    hann = torch.hann_window(
        ps + 2, periodic=False, dtype=avg.dtype, device=avg.device
    )[1:-1]
    win = (hann[:, None] * hann[None]).reshape(1, -1, 1)
    wgt = win * m_p * valid  # [1, P, L]
    den = F.fold(wgt, (H, W), ps, stride=st)[0, 0]
    inf = torch.tensor(float("inf"), dtype=avg.dtype, device=avg.device)
    a2 = avg.reshape(-1, 1, H, W)
    s2 = [s.reshape(-1, 1, H, W) for s in singles]
    out = torch.empty_like(a2)
    ar = torch.arange(n, device=avg.device)
    for b in range(0, len(a2), chunk):
        sl = slice(b, b + chunk)
        ap = torch.where(m_p, F.unfold(a2[sl], ps, stride=st), inf)  # [B, P, L]
        pool = torch.cat(
            [torch.where(m_p, F.unfold(s[sl], ps, stride=st), inf) for s in s2], 1
        )
        pool = pool.sort(1).values  # fluid values first
        rank = ap.argsort(1).argsort(1)
        idx = (rank[:, :, None] * n + ar[None, None, :, None]).flatten(
            1, 2
        )  # [B, P*n, L]
        mapped = pool.gather(1, idx).unflatten(1, (-1, n)).mean(2)
        mapped = torch.where(
            m_p, mapped, torch.zeros((), dtype=avg.dtype, device=avg.device)
        )
        num = F.fold(mapped * wgt, (H, W), ps, stride=st)
        out[sl] = torch.where(den > 0, num / den.clamp_min(1e-12), a2[sl])
    return out.reshape(avg.shape) * fl

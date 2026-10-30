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

# ruff: noqa: E741
import torch
import numpy as np


class QGPhysics:
    """
    Physical QG configuration matching 'Thermalizer' repo.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.NX = 64

        # --- 1. PHYSICAL CONSTANTS (SI) ---
        self.L = 1_000_000.0  # 1000 km
        self.H1 = 500.0
        self.H2 = 2000.0
        self.RD = 15000.0  # 15 km

        # --- 2. DENORMALIZATION CONSTANTS (From their util.py) ---
        # These are the standard deviations of the training data in Physical Units
        self.UPPER_STD = 8.6294e-06
        self.LOWER_STD = 1.1706e-06

        self.H_total = self.H1 + self.H2
        self.delta = self.H1 / self.H2

        # --- 3. PHYSICAL SPECTRAL GRID ---
        # freq = cycles per meter
        freq = torch.fft.fftfreq(self.NX, d=self.L / self.NX).to(device)
        # k = radians per meter (Physical: ~10^-5)
        k = freq * 2 * np.pi
        l = freq * 2 * np.pi

        KX, KY = torch.meshgrid(k, l, indexing="xy")
        self.K2 = KX**2 + KY**2

        # --- 4. INVERSION KERNEL ---
        kd2 = 1.0 / (self.RD**2)
        F1 = kd2 / (1 + self.delta)
        F2 = self.delta * F1

        det = (self.K2 + F1) * (self.K2 + F2) - F1 * F2
        det[0, 0] = 1.0

        self.inv_00 = -(self.K2 + F2) / det
        self.inv_01 = -F1 / det
        self.inv_10 = -F2 / det
        self.inv_11 = -(self.K2 + F1) / det

        # Zero mean flow
        self.inv_00[0, 0] = 0
        self.inv_01[0, 0] = 0
        self.inv_10[0, 0] = 0
        self.inv_11[0, 0] = 0

    def invert_pv_to_streamfunction(self, q):
        # Input q must be in PHYSICAL units (s^-1)
        is_batch = q.ndim == 4
        if not is_batch:
            q = q.unsqueeze(0)

        print("Layer 1 STD:", q[:, 0].std())
        print("Layer 2 STD:", q[:, 1].std())
        q_hat = torch.fft.fftn(q, dim=(-2, -1))

        psi1_hat = self.inv_00 * q_hat[:, 0] + self.inv_01 * q_hat[:, 1]
        psi2_hat = self.inv_10 * q_hat[:, 0] + self.inv_11 * q_hat[:, 1]

        psi_hat = torch.stack([psi1_hat, psi2_hat], dim=1)
        res = torch.fft.ifftn(psi_hat, dim=(-2, -1)).real

        if not is_batch:
            return res.squeeze(0)
        return res

    def invert_pv_to_streamfunction_unit(self, q):
        """
        PV -> streamfunction inversion ignoring physics, unit-variance only.
        q: (B, C, H, W)
        Returns psi same shape.
        """
        is_batch = q.ndim == 4
        if not is_batch:
            q = q.unsqueeze(0)  # add batch

        B, C, H, W = q.shape

        # FFT along spatial dims, default norm (norm=None) for Parseval consistency
        q_hat = torch.fft.fftn(q, dim=(-2, -1))

        # Build 2D wavenumber grid
        kx = torch.fft.fftfreq(H).to(q.device) * 2 * np.pi
        ky = torch.fft.fftfreq(W).to(q.device) * 2 * np.pi
        kx, ky = torch.meshgrid(kx, ky, indexing="ij")
        K2 = kx**2 + ky**2
        K2[0, 0] = 1.0  # avoid division by zero

        # Simple pseudo-inverse
        psi_hat = q_hat / (1.0 + K2)

        # Zero mean mode
        psi_hat[:, :, 0, 0] = 0.0

        # Back to spatial domain
        psi = torch.fft.ifftn(psi_hat, dim=(-2, -1)).real

        if not is_batch:
            psi = psi.squeeze(0)
        return psi

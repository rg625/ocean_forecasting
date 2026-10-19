# Architecture: conditional continuous-time Koopman autoencoder

This document defines the legacy formulation (as audited) and the redesigned CT-KAE
(`koopman_autoencoder/models/ct_kae.py`). It describes design and mathematics only. Which
variant performs best is decided by measurements in `EXPERIMENT_RESULTS.md`.

Notation:
- x_t ∈ R^{C×H×W}: TRA fields [v_x, v_y, p, ρ], C = 4, H×W = 64×128, z-scored per variable,
  obstacle cells zeroed.
- c: Mach number, the only physical parameter that varies in TRA. Re = 10⁴ is fixed.
- Δt = 0.1: benchmark frame spacing (raw data 0.05, stride 2).

## 1. Legacy formulation (reference; `models/autoencoder.py`)

    z_0   = ½ (E_hist(x_{t−1}) + E(x_t))                        single-query attention pooling, d = 128
    A(c)  = skew(W) + sym(D) + LoRA(c)                          D unconstrained
    z_k   = P₄(Δt A) z_{k−1}                                    one RK4 step per frame
    x̂_k   = D(z_k)                                              decoder ignores c

Properties, as measured in the audit:

| Property | Finding |
|---|---|
| Conditioning | Dead: LoRA U = V = 0 never trains, encoder `adaln` is unimplemented, decoder ignores c. Mach sensitivity is exactly 0 |
| Stability | Not guaranteed. QG checkpoint has 63/128 unstable modes. TRA has 16× transient growth |
| Shedding modes | Decay with an e-folding time of 36–41 steps (a learned hedge against phase error under an 8-step MSE objective) |
| Integrator | RK4 vs exp(Δt·A) differs by ≤ 1.6·10⁻⁴: numerically irrelevant |
| Latent targets | Encoded with a different map (present-only) from the rollout start (ENC-01) |

## 2. Redesigned CT-KAE

### 2.1 Model

    e        = MLP((c − c_ref) / c_scale)                        c_ref = 0.7, c_scale = 0.1
    z_0      = E_φ([x_{t−(n−1)s}, …, x_t], mask, coords; e)      delay window, n = 2
    z(t + τ) = μ(e) + exp(τ A(e)) (z_0 − μ(e))                   exact, any τ ≥ 0
    x̂(t + τ) = D_θ(z(t + τ); e) ⊙ mask

- **Encoder E_φ.** Inputs are the n frames stacked as channels, plus the obstacle mask and
  2 coordinate channels.
  - Three stages (64/128/256 channels), each with 2 residual blocks: GroupNorm → SiLU →
    3×3 conv, with FiLM(e) after the second norm.
  - A stride-2 conv follows each stage, giving an 8×16×256 map.
  - The map is flattened and passed through a linear layer to d = 256.
  - Flattening keeps spatial position. The legacy single-query attention pooling forced z
    to be a convex mixture of local features. There is no spectral norm and no dropout.
- **Decoder D_θ.** Mirrors the encoder: linear → 256×8×16, then three nearest-upsample +
  conv stages with FiLM(e) residual blocks, then GroupNorm → SiLU → conv to C channels,
  masked.
- **Fixed point μ(e).** μ = μ₀ + MLP(e), zero-initialised. It models the Mach-dependent mean
  flow: mean v_x rises monotonically from 0.44 at Ma 0.50 to 0.72 at Ma 0.89 in the training
  data.
- **Conditioning.** FiLM layers, the generator head and the μ head are all zero-initialised,
  so the model starts as the unconditional model. Every conditional path still receives
  gradient from step 1, because the head output enters linearly. The legacy U = V = 0 saddle
  (BUG-02) cannot occur; this is tested in `test_conditioning_receives_gradient_and_changes_output`.

### 2.2 Generators

| Name | A(c) | Propagation | Stability | Cost per sample |
|---|---|---|---|---|
| `block` (Candidates B + D) | blockdiag_j [[−γ_j(c), −ω_j(c)], [ω_j(c), −γ_j(c)]], γ_j = softplus(γ̃_j + Δγ_j(e)) ≥ 0, ω_j = ω̃_j + Δω_j(e) | closed form: e^{−γτ} R(ωτ) per block | ‖exp(τA)‖ ≤ 1 for all c; modes can be exactly neutral (γ → 0) | O(d) per lead; all leads in parallel |
| `hurwitz` (Candidate C) | J(c) − (LLᵀ + εI), J = skew(W + UVᵀ(e)) | `matrix_exp` (one per batch on a uniform grid, then products) | contractive in the Euclidean norm: no transient growth possible | O(d³) per sample |
| `dense` (Candidate A) | skew(W + UVᵀ(e)) + sym(W + UVᵀ(e)) + sym(D) | `matrix_exp` | none (the legacy form) | O(d³) |

**Design notes.**

- **Block form.** Any diagonalisable real A is similar to a real block-diagonal matrix of
  2×2 rotation-scalings. With a learned encoder and decoder, the change of basis is absorbed
  into E and D. The block form therefore loses no generality for diagonalisable dynamics; it
  only loses non-normal transient growth, which `dense` keeps.
- **Hurwitz form.** It deliberately forbids transient growth. The audit found 16× transient
  growth in the legacy TRA operator. Whether that growth was physical or an artefact is
  tested by comparing `hurwitz` with `dense` (pilots P1/P2), not assumed.
- **Frequency initialisation.** ω̃ is log-spaced over (0.05, 12) rad per time unit, which
  covers the observed shedding (ω ≈ 2.2–2.8) and its harmonics. γ̃ is log-uniform over
  (0.01, 0.3).
- **Precision.** Latent dynamics run in ≥ fp32 outside autocast. Encoder and decoder run
  in bf16.
- **Exactness.** Propagation is exact, so z(t₁ + t₂) = Φ(t₂)Φ(t₁)z (tested to 1e−9 in
  float64). Queries at arbitrary τ need no integrator, which supports irregular sampling
  and zero-shot Δt.

### 2.3 Training objective (`train_ctkae.py`)

For a batch of windows ending at t, drawn at stride s ∈ `strides`, with leads k = 1..K
(τ_k = k·s·0.05) and a random subset S of leads (always containing 1 and K, |S| = n_dec):

    L = w_rec  ‖D(z_0) − x_t‖²_w
      + w_pred mean_{k∈S} ‖D(z(τ_k)) − x_{t+ks}‖²_w
      + w_lat  mean_{k∈S} ‖z(τ_k) − sg[E(window ending at t+ks)]‖² / Var(sg[E])
      + w_amp  mean_{k∈S,j} (‖y_{k,j}‖ − ‖sg[y^E_{k,j}]‖)² / E‖y^E‖²      (block only; y = z − μ)

- ‖·‖_w is the mean squared error with channel weights ∝ σ_c². With these weights,
  normalised-space MSE equals the physical-unit MSE that the benchmark reports
  (`channel_weighting=phys`).
- **Horizon curriculum:** K grows linearly from `k_start` to `k_max` over the first
  `curriculum_frac` of training.
- **Latent consistency:** compares against stop-gradient targets encoded with the same
  window map that starts the rollout (fixing ENC-01). The stop-gradient prevents the encoder
  from making the targets easy, while reconstruction and prediction ground the latent in
  data.
- **Amplitude loss:** phase-invariant, acting on per-block radii ‖y_j‖. It targets the
  mean-regression mechanism directly. Under an MSE objective, an uncertain phase rewards
  shrinking the amplitude; the amplitude loss rewards keeping the true modal amplitude
  without fixing the phase.
- **Optimiser:** AdamW (lr 3e-4, β = (0.9, 0.99)), with no weight decay on generator, μ or
  1-D parameters. Linear warmup, then cosine decay to 5%. Gradient clipping at 1.0.
- **Validation:** held-out Ma = 0.60 and 0.75, removed from training and never a test case.
  Rollouts of 60 and 240 steps at Δt = 0.1, scored as MSE in physical units. Selection uses
  `val_mse60 + 0.25·val_mse240`. Test files are never read during training.

### 2.4 Inference

One encoder call on the input window, then **all** forecast leads in a single parallel
evaluation of exp(τ_k A)(z_0 − μ) (no sequential recurrence), then batched decoding.

Per-step costs:
- Latent cost: O(d) per lead for `block`, versus O(d²) per step for legacy RK4 matvecs.
- Decoder cost: dominates and is linear in the number of requested frames.

### 2.5 Complexity and size (base config)

24.7 M parameters. The encoder and decoder dominate: the flatten → linear layers alone hold
2 × 32768 × 256 ≈ 16.8 M parameters. The generator `block` has 2·d plus a 2-layer head.

Measured training throughput: 0.31–0.33 s/step at batch 16 with 8 decoded leads, on an
RTX 4090 shared with another job. Peak memory is 12.5 GB.

### 2.6 Trade-offs and limitations

- A linear generator cannot represent the quasi-periodic, broadband component of the wake.
  The shedding period wanders between about 23 and 40 frames across Mach, with sub-harmonic
  modulation. Pointwise skill must eventually decay, so long-horizon fidelity is judged by
  energy ratio and spectra, not only by MSE.
- The block form excludes non-normal transient growth. If `dense` beats `block` and
  `hurwitz`, that would indicate growth matters.
- Zero-shot claims are limited to Ma ∈ [0.50, 0.68] (the test cases). The training range
  is 0.53–0.89 with a gap at 0.64–0.68. Ma = 0.50–0.52 is mild extrapolation.

# Literature survey: post-processing forecast fields to lower error without blurring

**How references were checked.** I confirmed each reference by web search: authors, year, venue, and DOI or arXiv ID. Where a detail could not be confirmed, the entry says so. I did not read most full texts; claims about content come from abstracts and publisher or repository records.

**Abbreviations.**
- **Mbr**: the small member set (5 mixture decodes, 1 y-flip, 3 seeds).
- **Train?**: whether the method needs training. "Val-fit" means a few scalars are fitted on validation rollouts.
- **Cost**: implementation cost, L (low), M (medium) or H (high).
- **MSE / Sharp**: expected effect on MSE and on sharpness (LSiM and spectra).

---

## 0. Theory to anchor decisions

| Topic | Verified refs | Key content | Implication for you |
|---|---|---|---|
| Perception–distortion (PD) trade-off | Blau & Michaeli, CVPR 2018, [arXiv:1711.06077](https://arxiv.org/abs/1711.06077) | Some regions of the (distortion, perceptual-divergence) plane cannot be reached by any estimator. This holds for every distortion measure. | No post-processor can match the MSE of the ensemble mean and also match the true spectra. You can only choose a point on the curve. |
| DP curve in Wasserstein space | Freirich, Michaeli & Meir, NeurIPS 2021, [arXiv:2107.02555](https://arxiv.org/abs/2107.02555) | Under MSE with a W2 perception index, the DP function is quadratic. Optimal estimators lie on a W2 geodesic, built from the MMSE estimate and the perfect-perception estimate, which is a transport of the MMSE estimate. | Recipe: "best mean, then optimally transport it." For stationary Gaussian fields this transport is a per-wavenumber gain √(P_true/P_mean). Your spectral rescaling is that α=1 endpoint. The intermediate gains g_α(k) = 1 + α(√ratio − 1) trace the curve. (This Gaussian specialisation is my derivation from the paper's Gaussian results, not a quoted statement.) |
| Practical version: PMRF | Ohayon, Michaeli & Elad, ICLR 2025, [arXiv:2410.00418](https://arxiv.org/abs/2410.00418) | Predict the posterior mean, then use a rectified flow to transport it to the data distribution. It approaches minimum MSE among photo-realistic estimators. | A template for a learned "mean → sharp" refiner. |
| Scoring-function consistency | Gneiting 2011, JASA, [doi:10.1198/jasa.2011.r10138](https://doi.org/10.1198/jasa.2011.r10138) | MSE is consistent for the mean functional. | Any sharpening must cost some MSE relative to the true conditional mean. |
| Ensemble averaging acts as a filter | Surcel, Zawadzki & Yau 2014, MWR 142:1093–1105 (citation verified, **DOI not verified**) | Averaging removes the scales where members disagree, which gives a low-intensity bias. | Explains your loss of power at mid and high wavenumbers. |
| Phase versus amplitude | Oppenheim & Lim 1981, Proc. IEEE, [doi:10.1109/PROC.1981.12022](https://doi.org/10.1109/PROC.1981.12022) | Structure is carried mostly by phase. | The "mean amplitude + phase of the mean" approach cannot restore structure the mean's phases have already lost. |
| Double penalty / spatial verification | Gilleland et al. 2009, WAF, [doi:10.1175/2009WAF2222269.1](https://doi.org/10.1175/2009WAF2222269.1); Ebert 2008, Met. Apps, [doi:10.1002/met.25](https://doi.org/10.1002/met.25); Mass et al. 2002, BAMS, [doi:10.1175/1520-0477(2002)083<0407:DIHRPM>2.3.CO;2](https://doi.org/10.1175/1520-0477(2002)083%3C0407:DIHRPM%3E2.3.CO;2) | Displaced features are penalised twice, so gridpoint scores favour smooth forecasts. | Phase errors in the vortex shedding are displacement-type, so alignment-based methods target exactly this. **Not verified:** the Rossa, Nurmi & Ebert 2008 Springer chapter (no record found). |
| Blurring and spectral bias of ML models | Bonavita 2024, GRL, [doi:10.1029/2023GL107377](https://doi.org/10.1029/2023GL107377); Subich et al. 2025, ICML (AMSE), [arXiv:2501.19374](https://arxiv.org/abs/2501.19374); Lam et al. 2023, Science (GraphCast), [doi:10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336); Brenowitz et al. 2025, GRL, [arXiv:2401.15305](https://arxiv.org/abs/2401.15305); Ravuri et al. 2021, Nature, [doi:10.1038/s41586-021-03854-z](https://doi.org/10.1038/s41586-021-03854-z) | MSE training causes blurring that grows with lead time. AMSE separates the decorrelation error from the amplitude error; fine-tuning GraphCast with it raised effective resolution from 1,250 km to 160 km. Brenowitz et al. find multi-step losses increase dissipation. | **Only partly verified:** that GraphCast blurs with uncertainty is confirmed only through secondary sources that cite it; I could not retrieve the paper's own wording. |

---

## 1. NWP statistical post-processing

| Method | Verified refs | What it does | Train? | Fits your setting? | MSE / Sharp | Cost |
|---|---|---|---|---|---|---|
| MOS | Glahn & Lowry 1972, JAM 11:1203, [doi:10.1175/1520-0450(1972)011<1203:TUOMOS>2.0.CO;2](https://doi.org/10.1175/1520-0450(1972)011%3C1203:TUOMOS%3E2.0.CO;2) (DOI from an aggregator) | Regresses the target on model outputs. | Yes, (forecast, truth) pairs | Pointwise or per-mode regression on training rollouts | MSE↓, but blurs (it is a regression toward the mean) | L |
| EMOS/NGR | Gneiting et al. 2005, MWR 133:1098, [doi:10.1175/MWR2904.1](https://doi.org/10.1175/MWR2904.1) | Gaussian predictive distribution whose mean and variance are linear in the ensemble; fitted by minimum CRPS. | Yes | Its mean is just a weighted mean | MSE↓ / blurs | L |
| BMA | Raftery et al. 2005, MWR 133:1155, [doi:10.1175/MWR2906.1](https://doi.org/10.1175/MWR2906.1) | Mixture of member kernels with performance weights. | Yes | Weights for your 9 members | Point forecast blurs | L |
| Member-by-member (MBM) | Van Schaeybroeck & Vannitsem 2015, QJRMS 141:807, [doi:10.1002/qj.2397](https://doi.org/10.1002/qj.2397) | Corrects each member: shift plus scaling of the mean and of the deviations from the mean. Keeps member structure. | Yes (few params) | Yes | Each member stays sharp; MSE only lower if you then average | L |
| Superensemble / stacking | Krishnamurti et al. 1999, Science, [doi:10.1126/science.285.5433.1548](https://doi.org/10.1126/science.285.5433.1548) | Regression-weighted member combination; beat the plain ensemble mean. | Val-fit | Yes. Can be done per wavenumber band. | MSE↓ more than plain averaging / still blurs unless combined with §5 | L |
| Neural post-processing (distributional) | Rasp & Lerch 2018, MWR 146:3885, [doi:10.1175/MWR-D-18-0187.1](https://doi.org/10.1175/MWR-D-18-0187.1) | Neural network outputs distribution parameters. | Yes | Pointwise; little spatial structure | Calibration, not sharpness | M |
| Deep neural post-processing (spatial) | Grönquist et al. 2021, Phil. Trans. A 379:20200092, [doi:10.1098/rsta.2020.0092](https://doi.org/10.1098/rsta.2020.0092) | CNN corrects a reduced ensemble (+13% skill). | Yes (training rollouts) | Yes: input the member stack | Depends on loss: MSE blurs; add AMSE or a spectral loss | M |
| Permutation-invariant neural post-processing | Höhlein et al. 2024, AIES, [doi:10.1175/AIES-D-23-0070.1](https://doi.org/10.1175/AIES-D-23-0070.1) (DOI via arXiv record), [arXiv:2309.04452](https://arxiv.org/abs/2309.04452) | Treats members as an unordered set. | Yes | Good for your varying member types | Same as above | M |
| Review | Vannitsem et al. 2021, BAMS 102:E681, [doi:10.1175/BAMS-D-19-0308.1](https://doi.org/10.1175/BAMS-D-19-0308.1) | Overview of the field. | – | – | – | – |

## 2. Restoring spatial and multivariate structure

| Method | Verified refs | What it does | Fits your setting? / effect | Cost |
|---|---|---|---|---|
| Ensemble copula coupling (ECC) | Schefzik, Thorarinsdottir & Gneiting 2013, Stat. Sci. 28:616, [doi:10.1214/13-STS443](https://doi.org/10.1214/13-STS443) | Takes quantiles from calibrated marginals and reorders them by the raw ensemble's rank structure. | Produces a sharp *ensemble*, not a sharp point forecast. With 9 members the quantiles are coarse. Mainly useful after MBM or EMOS if you want a set of sharp members. | L |
| Schaake shuffle | Clark et al. 2004, J. Hydromet. 5:243, [doi:10.1175/1525-7541(2004)005<0243:TSSAMF>2.0.CO;2](https://doi.org/10.1175/1525-7541(2004)005%3C0243:TSSAMF%3E2.0.CO;2) | Same idea, with ranks taken from historical (training) fields. | As ECC. Could impose realistic rank structure from training snapshots. | L |

## 3. Sharp ensemble summaries

| Method | Verified refs | What it does | Train? | Fits your setting? | MSE / Sharp | Cost |
|---|---|---|---|---|---|---|
| Probability-matched mean (PMM) | Ebert 2001, MWR 129:2461, [doi:10.1175/1520-0493(2001)129<2461:AOAPMS>2.0.CO;2](https://doi.org/10.1175/1520-0493(2001)129%3C2461:AOAPMS%3E2.0.CO;2) | Keeps the mean's spatial ranks and replaces its values with the pooled member distribution. | No | This is your global quantile mapping | Restores the one-point PDF, not the spectrum; MSE↑ (an inflation effect, see §8) | L |
| Localized PMM (LPM), point-wise and patch-wise | Clark 2017, WAF, [doi:10.1175/WAF-D-16-0199.1](https://doi.org/10.1175/WAF-D-16-0199.1); Snook, Kong & Brewster 2019, WAF 34:781, [doi:10.1175/WAF-D-18-0155.1](https://doi.org/10.1175/WAF-D-18-0155.1); Snook et al. 2020, GRL 47, [doi:10.1029/2020GL087839](https://doi.org/10.1029/2020GL087839) | PMM within local windows. The 2020 paper reports it beats both PMM and the simple mean on FSS and variance spectra. | No | Yes. Apply per field on overlapping patches. | Better spectra than PMM; MSE close to the mean's in the NWP studies (precipitation, non-negative fields) | L |
| Spatial Aligned Mean (SAM), with SAM-LPM | Lee, Brewster, Snook, Spencer & Park 2025, WAF, [doi:10.1175/WAF-D-23-0229.1](https://doi.org/10.1175/WAF-D-23-0229.1) | Multiscale alignment of every member to a common location, then average, then LPM. | No | Directly analogous; a generalisation of your integer x-shift to dense multiscale warps | MSE↓ and structure kept | M |
| Feature-oriented mean | Zhang et al. 2021, WAF 36:1945, [doi:10.1175/WAF-D-21-0062.1](https://doi.org/10.1175/WAF-D-21-0062.1); Zhang & Li 2024, Remote Sens., [doi:10.3390/rs16091596](https://doi.org/10.3390/rs16091596) | Average feature positions, then amplitudes. The 2024 paper reports lower RMSE than the arithmetic mean. | No | Vortex-street phase is a natural "feature" | MSE↓ / sharp | M |
| Field alignment | Ravela, Emanuel & McLaughlin 2007, Physica D 230:127, [doi:10.1016/j.physd.2006.09.035](https://doi.org/10.1016/j.physd.2006.09.035); Brewster 2003 Part I, MWR 131:480, [doi:10.1175/1520-0493(2003)131<0480:PCDAAA>2.0.CO;2](https://doi.org/10.1175/1520-0493(2003)131%3C0480:PCDAAA%3E2.0.CO;2) (Part II, MWR 131:493–507, **DOI not verified**); Nehrkorn et al. 2015, MWR 143:1368, [doi:10.1175/MWR-D-14-00127.1](https://doi.org/10.1175/MWR-D-14-00127.1) | Solve for a smooth displacement field, then correct amplitude. | No (variational solve) | Yes. Gives the displacement machinery for an aligned mean. | – | M |
| Morphing | Beezley & Mandel 2008, Tellus A 60:131, [doi:10.1111/j.1600-0870.2007.00275.x](https://doi.org/10.1111/j.1600-0870.2007.00275.x) | Represents members as (warp, residual) relative to a reference, then averages in that space. | No | Same use as field alignment | – | M |
| Wasserstein barycenter / displacement interpolation | McCann 1997, Adv. Math. 128:153, [doi:10.1006/aima.1997.1634](https://doi.org/10.1006/aima.1997.1634); Agueh & Carlier 2011, SIAM J. Math. Anal. 43:904, [doi:10.1137/100805741](https://doi.org/10.1137/100805741); Bonneel et al. 2011, ACM TOG 30(6), [doi:10.1145/2070781.2024192](https://doi.org/10.1145/2070781.2024192); Solomon et al. 2015, ACM TOG 34(4), [doi:10.1145/2766963](https://doi.org/10.1145/2766963); Peyré & Cuturi 2019, [doi:10.1561/2200000073](https://doi.org/10.1561/2200000073); geoscience use: Tamang et al. 2021, NPG 28:295, [doi:10.5194/npg-28-295-2021](https://doi.org/10.5194/npg-28-295-2021) | A mass-preserving "average" that moves structures rather than superimposing them. | No | Needs positive, mass-normalised fields. Density works; vorticity and velocity must be split into positive and negative parts, or use unbalanced OT. Fast on 64×128 with convolutional Sinkhorn. | Sharp. MSE often above the Euclidean mean, but can be below a single member. | M |

**Note on the "Nehrkorn et al. 2014" item you listed.** I could not find the Tellus paper "Feature calibration and alignment to represent model forecast errors". That title belongs to Nehrkorn, Hoffman, Grassotti & Louis 2003, QJRMS 129:195–218 (**DOI not verified**). The 2015 MWR paper above is the verified substitute.

## 4. Spectral and stochastic sharpening

| Method | Verified refs | What it does | Train? | MSE / Sharp | Cost |
|---|---|---|---|---|---|
| Scale-dependent blending (S-PROG / STEPS / pySTEPS) | Seed 2003, J. Appl. Meteor. 42:381–388 (citation verified, **DOI not verified**); Bowler, Pierce & Seed 2006, QJRMS 132:2127, [doi:10.1256/qj.04.100](https://doi.org/10.1256/qj.04.100); Pulkkinen et al. 2019, GMD 12:4185, [doi:10.5194/gmd-12-4185-2019](https://doi.org/10.5194/gmd-12-4185-2019) | Splits fields into a spectral cascade, weights each level by its skill, and fills lost variance with correlated noise. | Val-fit (per-level weights) | Per-band weights between the mean and a member can be MSE-optimal per band while keeping the power. Noise injection raises MSE. | L |
| Spectral nudging | von Storch, Langenberg & Feser 2000, MWR 128:3664, [doi:10.1175/1520-0493(2000)128<3664:ASNTFD>2.0.CO;2](https://doi.org/10.1175/1520-0493(2000)128%3C3664:ASNTFD%3E2.0.CO;2) | Relaxes the large scales toward a driver while leaving small scales free. | No | Use the ensemble mean as the driver for low k and keep one member's high k: MSE↓ at large scales, small-scale spectrum intact | L |
| Stochastic backscatter | Leith 1990, Phys. Fluids A 2:297, [doi:10.1063/1.857779](https://doi.org/10.1063/1.857779); Shutts 2005, QJRMS 131:3079, [doi:10.1256/qj.04.106](https://doi.org/10.1256/qj.04.106) | Injects energy at unresolved or lost scales. | No | Restores power; adds about the injected variance to the MSE | L |
| Randomisation instead of inflation | von Storch 1999, J. Climate 12:3505 (**DOI not verified**) | Argues against variance inflation; add noise instead. | – | Theoretical caution for §8 | – |
| Unsharp masking / Wiener deconvolution, spectral rescaling | No specific paper needed | A deterministic gain per wavenumber. | Val-fit | This is the Gaussian-OT direction in §0. Use partial α, not α=1. | L |

## 5. Generative refinement

| Method | Verified refs | What it does | Train? | Fits your setting? | MSE / Sharp | Cost |
|---|---|---|---|---|---|---|
| SDEdit | Meng et al., ICLR 2022, [arXiv:2108.01073](https://arxiv.org/abs/2108.01073) | Add noise to a guide image up to level t₀, then run the reverse diffusion. No task-specific training. | No | **Yes, with ACDM**: noise the mean (or a member) to t₀ and denoise conditioned on the previous states. | t₀ sets the PD point: small t₀ keeps MSE near the input, larger t₀ restores the spectrum | M (inference) |
| Diffusion refinement of flow surrogates | Shu, Li & Barati Farimani 2023, JCP 478:111972, [doi:10.1016/j.jcp.2023.111972](https://doi.org/10.1016/j.jcp.2023.111972); Oommen et al. 2024, [arXiv:2409.08477](https://arxiv.org/abs/2409.08477) | Shu et al.: SDEdit-like reconstruction of 2D turbulence from low-fidelity input, no retraining. Oommen et al.: diffusion conditioned on a neural-operator output restores the energy spectrum. | Oommen: yes | Closest precedent to "Koopman AE + ACDM refiner" | Spectra↑; MSE reported comparable | M–H |
| PDE-Refiner | Lippe et al., NeurIPS 2023, [arXiv:2308.05732](https://arxiv.org/abs/2308.05732) | Multistep denoising refinement that models all frequency bands. | Yes (it is the surrogate itself) | Would require retraining the surrogate; mainly conceptual for you | – | H |
| Residual / corrective diffusion | Mardani et al. 2025 (CorrDiff), Commun. Earth Environ., [doi:10.1038/s43247-025-02042-5](https://doi.org/10.1038/s43247-025-02042-5), [arXiv:2309.15214](https://arxiv.org/abs/2309.15214); Couairon et al. 2024 (ArchesWeatherGen), [arXiv:2412.12971](https://arxiv.org/abs/2412.12971); Yu et al., CVPR 2024 (DiffCast), [arXiv:2312.06734](https://arxiv.org/abs/2312.06734) | A deterministic model predicts the mean; diffusion or flow matching models the residual. ArchesWeatherGen explicitly targets the smoothing of deterministic forecasts. | Yes, (mean, truth − mean) pairs | Yes: train on training-set rollouts of your Koopman AE | A sample is sharp; MSE ≈ MMSE + residual variance | H |
| Debias-then-sample | Wan et al., NeurIPS 2023, [arXiv:2305.15618](https://arxiv.org/abs/2305.15618) | An OT map debiases, then a diffusion model samples conditionally. | Yes | Template for OT followed by ACDM | – | H |
| Generative ensembles | Price et al. 2025 (GenCast), Nature 637:84, [doi:10.1038/s41586-024-08252-9](https://doi.org/10.1038/s41586-024-08252-9); Li et al. 2024 (SEEDS), Sci. Adv. 10:eadk4489, [doi:10.1126/sciadv.adk4489](https://doi.org/10.1126/sciadv.adk4489) (published author list only partly confirmed); Molinaro et al. 2024 (GenCFD), [arXiv:2409.18359](https://arxiv.org/abs/2409.18359) | SEEDS generates large ensembles from 1–2 seed forecasts. GenCFD shows deterministic operators regress to the mean. | Yes | Inflating your 9 members, then summarising with §3 | – | H |
| GAN post-processing | Harris et al. 2022, JAMES, [doi:10.1029/2022MS003120](https://doi.org/10.1029/2022MS003120) (authors Harris, McRae, Chantry, Dueben & Palmer); Hess et al. 2022, Nat. Mach. Intell. 4:828, [doi:10.1038/s42256-022-00540-1](https://doi.org/10.1038/s42256-022-00540-1) | Adversarial correction of spatial and spectral statistics. | Yes | Possible, but more fragile than diffusion | – | H |
| Image-restoration precedent | Whang et al., CVPR 2022, [doi:10.1109/CVPR52688.2022.01581](https://doi.org/10.1109/CVPR52688.2022.01581) | A deterministic predictor plus a stochastic diffusion refiner of its output. | Yes | Same structure as surrogate + ACDM | – | H |

## 6. Analog methods

| Method | Verified refs | What it does | Fits your setting? | Cost |
|---|---|---|---|---|
| Analog ensemble (AnEn) | Delle Monache et al. 2013, MWR 141:3498, [doi:10.1175/MWR-D-12-00281.1](https://doi.org/10.1175/MWR-D-12-00281.1); Hamill & Whitaker 2006, MWR 134:3209, [doi:10.1175/MWR3237.1](https://doi.org/10.1175/MWR3237.1) | Find past forecasts that resemble the current one and use their verifying truths as members. | Yes, using a library of training-set rollouts paired with truth. The analog truths are real, sharp fields. Their mean blurs again, so use one analog's small scales (spectral transplant) or LPM. | M |

## 7. Bias correction and distribution mapping

| Method | Verified refs | What it does | Fits your setting? / effect | Cost |
|---|---|---|---|---|
| Quantile mapping, and its inflation caveat | Maraun 2013, J. Climate 26:2137, [doi:10.1175/JCLI-D-12-00821.1](https://doi.org/10.1175/JCLI-D-12-00821.1) | Shows QM applied across a scale or variance mismatch is effectively variance inflation, which increases error. | Explains why your global QM of the mean raised MSE | L |
| Quantile delta mapping (QDM) | Cannon, Sobie & Murdock 2015, J. Climate 28:6938, [doi:10.1175/JCLI-D-14-00754.1](https://doi.org/10.1175/JCLI-D-14-00754.1) | Preserves the model's relative changes per quantile. | Marginal statistics only | L |
| Multivariate bias correction (MBCn) | Cannon 2018, Clim. Dyn. 50:31, [doi:10.1007/s00382-017-3580-6](https://doi.org/10.1007/s00382-017-3580-6) | N-dimensional PDF transfer using random rotations plus 1-D QM. | Could transfer the joint distribution of the 4 fields or of local patches. Still a PDF fix, not a fix for spatial phase. | L–M |

## 8. Time-lagged ensembles and symmetry test-time augmentation

| Method | Verified refs | Fits your setting? | Cost |
|---|---|---|---|
| Lagged average forecasting | Hoffman & Kalnay 1983, Tellus A 35:100, [doi:10.3402/tellusa.v35i2.11425](https://doi.org/10.3402/tellusa.v35i2.11425) (an alternative Wiley DOI also appears in some records); Brenowitz et al. 2025 (above) | Extra members from earlier start times, if your data provides earlier frames. These members are independent and decorrelated at small scales, so their mean blurs, the same as yours. | L |
| Frame averaging / test-time augmentation | Puny et al., ICLR 2022, [arXiv:2110.03336](https://arxiv.org/abs/2110.03336); Shanmugam et al., ICCV 2021, [doi:10.1109/ICCV48922.2021.00125](https://doi.org/10.1109/ICCV48922.2021.00125) | Your y-flip member is a case of this. Shanmugam et al. find learned aggregation beats a plain mean. | L |

---

## Ranked shortlist for your goal

Goal: MSE below a single deterministic forecast, with LSiM and spectra no worse.

1. **Per-wavenumber blending of the mean and a single member, STEPS-style** (Bowler et al. 2006; von Storch et al. 2000).
   - x̂(k) = w(k)·mean(k) + (1 − w(k))·member(k). Choose w(k) on validation to minimise MSE subject to the band power being at least the member's.
   - Large scales then come from the low-error mean; small scales from a member, so phase-coherent structure is kept.
   - Cost L, no training, and it directly targets your two metrics. Start here.
2. **Partial optimal-transport sharpening along the DP curve** (Freirich et al. 2021).
   - Apply g_α(k) = 1 + α(√(P_mbr/P_mean) − 1) and sweep α, or α per band. Your α=1 test is only the endpoint.
   - Theory says the optimal compromise lies in between, and every sharpened point costs a predictable amount of MSE.
   - Cost L. Combine with #1.
3. **Dense-alignment mean: SAM, feature-oriented mean, or field alignment** (Lee et al. 2025; Zhang et al. 2021; Ravela et al. 2007; Beezley & Mandel 2008).
   - Use optical flow or diffeomorphic registration of each member to a reference (the medoid), average the warped fields, then apply the average warp.
   - Shedding-phase and convection errors are displacement-type, so this attacks the double penalty directly. Integer x-shift is too restrictive.
   - Cost M, no training.
4. **SDEdit with your trained ACDM, starting from the (aligned or blended) mean** (Meng et al. 2022; Shu et al. 2023; Oommen et al. 2024; Ohayon et al. 2025).
   - Noise to a small t₀ and denoise conditioned on the history; sweep t₀.
   - No training needed. Expect spectra and LSiM to recover, with MSE between the mean's and a member's. Do not average the samples.
   - Cost M (inference only).
5. **Learned combiner with a sharpness-aware loss** (Subich et al. 2025 AMSE; Grönquist et al. 2021; Höhlein et al. 2024).
   - A small U-Net or set network takes the 9 members (plus mean and spread) and is trained on training-set rollouts with AMSE or MSE plus a spectral term.
   - This is the most direct way to learn the MSE-versus-spectrum optimum for your data. Cost M.
6. **Localized PMM, patch-wise** (Clark 2017; Snook et al. 2020).
   - Replaces your failing global QM, and was shown to improve the variance spectrum over both PMM and the mean.
   - Cost L. Signed fields may need care; the published evidence is for non-negative precipitation.
7. **Member-by-member correction plus regression-weighted (superensemble) combination**, optionally per band (Van Schaeybroeck & Vannitsem 2015; Krishnamurti et al. 1999).
   - Cheap weights that beat an equal mean on MSE. Feed the result into #1 or #2.
   - Cost L.
8. **Residual diffusion or PMRF-style flow trained on (mean → truth)** (CorrDiff; ArchesWeatherGen; Ohayon et al. 2025).
   - The most principled route to "min-MSE subject to realism", but it needs new training.
   - Cost H. Do this if #4 shows promise but is limited by ACDM being conditioned on true histories.
9. **Wasserstein barycenter of members** (Solomon et al. 2015; Agueh & Carlier 2011), applied to density or to split positive and negative vorticity.
   - Sharp by construction. Its MSE gain over a member is uncertain.
   - Cost M.
10. **Analog spectral transplant**: training-set analogs supply the small scales (Delle Monache et al. 2013).
    - Sharp and physically realistic fine scales, but MSE at small scales is no better than a member's.
    - Cost M.

**Not recommended as primary:** plain noise injection or backscatter (restores the spectrum at a direct MSE cost; von Storch 1999, Maraun 2013); ECC or Schaake shuffle (they produce sharp ensembles, not point forecasts); more lagged or symmetry members alone (more averaging, more blur).

**Unverified items, flagged in the tables:** DOIs for Seed 2003, Surcel et al. 2014, von Storch 1999, Brewster 2003 Part II and Nehrkorn et al. 2003; the Rossa et al. 2008 chapter (not found); GraphCast's own blurring statement (secondary sources only); the full SEEDS author list. All other references were checked by web search, with their DOI or arXiv ID given above.

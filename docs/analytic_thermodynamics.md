# The Analytic Thermodynamics Fix

*Shipped May 14, 2026. This is the technical novelty of the Meridian × Gemma 4 system.*

---

## TL;DR

Van't Hoff (ΔG = −RT·ln(10)·pKd at 298 K) and the Gibbs equation (ΔG = ΔH + mTΔS) are **exact thermodynamic identities**, not relationships to be learned from data. The earlier Meridian training loss had z-normalized the thermodynamic prediction heads independently per batch, which stripped absolute scale and broke physical consistency. We replaced the learned ΔG and mTΔS heads with their analytic forms at inference. ΔH is the only remaining learned thermodynamic quantity. The Gemma 4 assistant can now reason about *why* a binding is favorable with full thermodynamic self-consistency.

## How we caught it

During a routine probe of model outputs across 78,384 predictions (16 hard targets × 4,899 generated SMILES), we ran consistency checks against the thermodynamic identities:

| Identity | Expected | Observed | Status |
|---|---|---|---|
| ΔG vs −RT·ln(10)·pKd (Van't Hoff) | r = 1.0, slope = −1.364 | r = 0.11, slope = −0.36 | **BROKEN** |
| ΔG vs (ΔH + mTΔS) (Gibbs) | r = 1.0, RMSE = 0 | r = −0.44, RMSE ≈ 4 kcal/mol | **ANTI-CORRELATED** |
| k_on, k_off (kinetic) | finite, monotonic | identically zero across all 78,384 | **DEAD HEAD** |

The slopes told us *something* was happening (rankings correlated), but the absolute relationships were destroyed. That's a signature of independent z-normalization at training time.

## Root cause

From the original training script (`train_v_clean_005c.py`, lines 418–421):

```python
loss_th = (F.mse_loss(zn(pred_dG),   zn(lbl_dG))
         + F.mse_loss(zn(pred_dH),   zn(lbl_dH))
         + F.mse_loss(zn(pred_mTdS), zn(lbl_mTdS))) / 3
```

where `zn(x) = (x - x.mean()) / (x.std() + 1e-6)`. The z-normalization is applied to **both** predictions and labels *per batch*. That means the loss only penalizes deviations in *ranking* within the batch — it gives the model zero gradient signal about absolute kcal/mol values. Each head learned its own internal scale that drifted away from the others.

The pKd loss was *not* z-normalized (line 322: `F.mse_loss(pkd_pred, pkd_lbl)`), which is why pKd predictions are honestly calibrated. The thermodynamic heads were the bug.

We verified directly: the model's *raw* ΔG output ranges only `[−0.77, +0.5]` kcal/mol on PDBbind test, where physics requires `[−15, −3]`. A post-hoc `thermo_calibration.json` was scaling raw_ΔG by 15.91× to make it look reasonable downstream — a calibration band-aid over a structurally too-narrow output range.

## Why we didn't just fine-tune

We tried first. A label-free consistency fine-tune (added L_vanthoff and L_gibbs losses on top of pKd, with pKd detached so consistency can't degrade the deliverable) produced:

| Step | pKd Pearson | ΔGibbs r | ΔVanHoff r |
|---|---|---|---|
| 0 (baseline) | 0.7714 | −0.109 | 0.170 |
| 100 | 0.7660 (−0.005) | **+0.339** | **+0.501** |
| 200 | 0.7436 (−0.028, aborted) | +0.510 | +0.487 |

We could get partial consistency at small pKd cost, but never full consistency. Reason: the model's ΔG head was structurally a normalized scoring head, not a kcal/mol head. Making it match Van't Hoff slope required expanding its output range by ~8×, which propagated aggressive gradients through the shared encoder and degraded the calibrated pKd — the actual deliverable.

## The fix

Instead of teaching the model thermodynamic identities it should already know, we *enforce them by construction* at inference. The Gemma 4 tool returns:

```python
RT_LN10 = 0.5925 * np.log(10)   # 1.3643 kcal/mol at 298 K

# Van't Hoff is an exact thermodynamic identity. Use it.
dG_kcal   = -RT_LN10 * pkd_calibrated     # always exact

# dH is the only learned thermodynamic quantity (linear calibration
# from model's enthalpy head, anchored to n=17 ITC test set,
# MAE 0.61 kcal/mol).
dH_kcal   = cal['dH']['a'] * model_dH_raw + cal['dH']['b']

# mTdS follows from the Gibbs equation. Use it.
mTdS_kcal = dG_kcal - dH_kcal             # always exact
```

Verification on the same 4,899-molecule probe (PfDHODH):

| Identity | After fix |
|---|---|
| Van't Hoff r | **1.000000** |
| Van't Hoff slope | **−1.364282** (matches RT·ln(10) to 6 digits) |
| Gibbs r | **1.000000** |
| Gibbs RMSE | **0.000000 kcal/mol** |
| pKd Pearson vs baseline | 1.0000 (unchanged) |

Physical ranges sanity check (PfDHODH, n=4899):
- pKd: `[2.93, 9.06]` → ΔG: `[−12.36, −4.00]` kcal/mol ✅
- ΔH: `[−13.59, −5.29]` kcal/mol (calibrated model output) ✅
- mTΔS: `[−4.02, +5.85]` kcal/mol (mix of entropy-favored and entropy-penalized binding, physically reasonable) ✅

## Why this matters for Gemma 4

Gemma 4 is now reasoning over a thermodynamic decomposition that **cannot violate first-law physics**. When the assistant says "the binding is enthalpy-driven, ΔH = −9.36 and mTΔS = −2.58," the user can be confident that ΔG = −9.36 + −2.58 = −11.94, which is exactly what −RT·ln(10)·pKd produces for the reported pKd of 8.75. Numbers add up because the architecture guarantees it.

This is the difference between a hallucinating chatbot and a calibrated decision support tool.

## What's still learned, and why

ΔH — the change in enthalpy on binding — is genuinely empirical. It depends on the specific molecular interactions (H-bonds, electrostatics, dispersion, water displacement) and cannot be derived from pKd alone. So the model retains an ΔH prediction head with a linear post-hoc calibration fit on a small ITC test set (n=17, MAE 0.61 kcal/mol on the held-out portion).

Known limitation: the underlying ITC dataset has Gibbs-residual quality issues across heterogeneous source measurements. Only ~17% of source entries satisfy ΔG ≈ ΔH + mTΔS to within 1 kcal/mol — likely because individual measurements were taken at different temperatures or pH conditions. We have a planned expansion to ~200 entries via cross-domain ITC mapping (132 protein-ligand entries in `crossdomain_thermo_v2`), pending SELFIES → v33b2 index mapping work.

## Generalizing: physics-as-constraint vs physics-as-residual

The original Meridian thesis is *physics-as-residual-input*: the model learns corrections to known physics rather than re-learning physics from scratch. The thermodynamic fix is the dual statement: *physics-as-architectural-constraint*. Where a relationship is an exact identity (Van't Hoff at fixed T, Gibbs at thermal equilibrium), enforce it in the architecture, not the loss.

This generalizes to other domains. Anywhere a model has multiple output heads coupled by conservation laws or exact identities (mass balance, energy conservation, charge neutrality, detailed balance, Onsager relations), the same pattern applies: *don't teach the model the identity. Encode it.*
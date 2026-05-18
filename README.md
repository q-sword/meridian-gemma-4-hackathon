# Meridian × Gemma 4 — Physics-Informed Drug Binding Assistant

*Submission to the Gemma 4 Good Hackathon — Health & Sciences track*

**Built by [Adrian Sword](https://github.com/q-sword) and Hanya — Sword Labs**

---

## What this is

A clinical decision-support assistant for drug-target binding analysis, demonstrated on antimalarial drugs. It wraps **Meridian v_clean_005c** — a 32.4M-parameter physics-informed binding affinity model (PDBbind Pearson 0.7773) — behind **Gemma 4 e4b** using native function calling.

Unlike generic LLM "explain this molecule" tools, this system **refuses to guess values**. Every binding number it reports comes from a tool call to a calibrated model, with explicit 80% confidence intervals and provenance flags ("this prediction is anchored to a real measurement" vs. "this is ESM-similarity extrapolated").

**The technical novelty: thermodynamic consistency by architectural construction.** Van't Hoff (ΔG = −RT·ln(10)·pKd) and the Gibbs equation (ΔG = ΔH + mTΔS) are *exact thermodynamic identities*, not things to learn. We enforce them analytically at inference. The model's only learned thermodynamic quantity is ΔH; ΔG and mTΔS are derived. This was the [outcome of a bug we caught on May 14](docs/analytic_thermodynamics.md) — the original training z-normalized loss had stripped absolute scale from the thermo heads.

## Watch the 3-minute demo

📺 **[YouTube — Meridian × Gemma 4 Demo](https://youtu.be/PLACEHOLDER_REPLACE_BEFORE_SUBMIT)**

## What the system does

User asks Gemma 4 in natural language:

> *"What is the predicted binding affinity of DSM265 against PfDHODH? Is the binding entropy-driven or enthalpy-driven?"*

Gemma 4 → decides to call `predict_drug_binding(drug_name="DSM265", target_uniprot="Q08210")` → Meridian returns a rich JSON bundle (pKd, 80% CI, ΔG, ΔH, mTΔS, calibration provenance, structural data flags, Lipinski) → Gemma 4 writes:

> **pKd (calibrated):** 8.75 (80% CI: 8.05–9.45) → **Kd ≈ 1.77 nM** (CI 0.35–8.85 nM)
>
> **ΔG:** −11.94 kcal/mol
> **ΔH:** −9.36 kcal/mol
> **mTΔS:** −2.58 kcal/mol
>
> The binding is predominantly **enthalpy-driven** — the large negative ΔH suggests favorable specific molecular interactions (hydrogen bonding, electrostatics) stabilizing the drug-target complex. The modest negative entropy term is consistent with hydrophobic-pocket binding.
>
> **⚠ Calibration:** This prediction is **ESM-similarity extrapolated** (anchored to KIT, similarity 0.911). PfDHODH lacks a direct anchor in our training corpus — interpret accordingly.
>
> *This analysis is for research context only and is not clinical advice.*

That's the *entire* user experience. No hallucinated numbers, no marketing language, explicit uncertainty.

## Why this matters (Health & Sciences track)

Neglected diseases — malaria, Chagas, TB — get a tiny fraction of the pharma R&D budget. Computational triage tools are how academic and non-profit labs punch above their weight. But generic LLMs hallucinate binding numbers, and standalone affinity predictors require the user to interpret raw outputs.

This system fuses them: Gemma 4 handles the natural-language interface and clinical-voice translation; Meridian provides the calibrated numbers with honest uncertainty; thermodynamic decomposition explains *why* a binding is favorable (enthalpy vs entropy), which guides medicinal chemistry decisions about hydrogen-bond donors, hydrophobic pockets, etc.

We demo three antimalarial drugs in the curated panel: **DSM265** (PfDHODH inhibitor), **Atovaquone** (PfCytB), **Pyrimethamine** (PfDHFR).

## Quick start

**Prerequisites:**
- Python 3.10+
- Ollama with `gemma4:e4b` pulled (`ollama pull gemma4:e4b`)
- For live mode: access to a Meridian inference endpoint (see [Server section](#server))
- For offline/cached mode: just clone and run — pre-recorded predictions ship in `demo/cached_responses.json`

**Run the demo (cached, no model server needed):**

```bash
git clone https://github.com/q-sword/meridian-gemma-4-hackathon
cd meridian-gemma-4-hackathon
pip install -r requirements.txt
export MERIDIAN_MODE=cached
python demo/gemma_meridian_demo.py --demo
```

**Run the demo (live, requires Meridian server):**

```bash
export MERIDIAN_URL=http://127.0.0.1:7891    # your Meridian inference endpoint
python demo/gemma_meridian_demo.py --query "Brief: how tight does Imatinib bind BCR-ABL?"
```

## Architecture

```
   ┌──────────┐    natural language query   ┌─────────────────┐
   │   User   │ ───────────────────────────▶│   Gemma 4 e4b   │
   └──────────┘                              │ (via Ollama)    │
        ▲                                    └─────────────────┘
        │                                            │
        │  clinical-voice answer with                │ tool_call:
        │  uncertainty, thermo profile,              │ predict_drug_binding(
        │  provenance, safety caveats                │   drug_name, uniprot
        │                                            │ )
        │                                            ▼
        │                                    ┌─────────────────┐
        │                                    │ Meridian HTTP   │
        │                                    │ server (FastAPI │
        │                                    │  stdlib)        │
        │                                    └─────────────────┘
        │                                            │
        │                                            ▼
        │                                    ┌─────────────────┐
        │                                    │ Meridian        │
        │                                    │ v_clean_005c    │
        │                                    │  (32.4M params) │
        │                                    │ + SmartCalibrat │
        │                                    │   (per-target)  │
        │                                    │ + analytic      │
        │                                    │   Van't Hoff +  │
        │                                    │   Gibbs         │
        │                                    └─────────────────┘
        │                                            │
        └────────────────────────────────────────────┘
                  rich JSON bundle: pKd, 80% CI in pKd + nM,
                  ΔG/ΔH/mTΔS (consistent by construction),
                  calibration_source, structural data flags,
                  Lipinski/MW/logP/TPSA
```

More detail: [`docs/architecture.md`](docs/architecture.md) and [`docs/analytic_thermodynamics.md`](docs/analytic_thermodynamics.md).

## Honest scope statement

**What's open in this repo:**
- Gemma 4 integration (system prompt, tool definitions, REPL/one-shot/demo modes)
- Meridian HTTP server wrapper (stdlib `http.server`)
- Cached prediction responses for offline reproducibility
- Architecture and design notes

**What's not in this repo:**
- Meridian model weights (~130MB, IP-protected — contact Sword Labs)
- Training data and full training pipeline (months of curation work)
- The MeridianAPI inference library (Sword Labs internal, ~700 lines)

The video demonstrates the system running live against the production Meridian backend. Cached responses in this repo let judges and contributors verify the Gemma 4 integration without needing model access.

## Status

**Production model**: Meridian v_clean_005c — 32.4M params, trained on a curated subset of BindingDB + ChEMBL + PDBbind (~1.2M binding measurements). PDBbind Pearson 0.7773 on held-out test set. SmartCalibrator with 7 target-family anchors covering ~600 measured drug-target pairs.

**Analytic thermodynamic fix**: shipped May 14, 2026. Replaces the previously z-normalized thermo training loss with analytic Van't Hoff at inference. ΔG, mTΔS now consistent with calibrated pKd to floating-point precision.

**Validation**: PfDHODH zero-shot DSM265 prediction (1.77 nM, no direct anchor) is within an order of magnitude of the real measured Kd (~12 nM). Imatinib → BCR-ABL prediction (10 nM, anchored) matches the FDA label.

## Limitations and roadmap

- Curated drug panel is currently 17 drugs. Arbitrary SMILES work via `predict_smiles_binding` but quality drops for chemistry far from the training distribution.
- Only 7 of ~600 targets have direct anchor calibrations. Others fall back to ESM-similarity extrapolation (the system flags this).
- ΔH calibration test set is n=17 — we have a known data-quality issue with Gibbs residuals in the source ITC corpus and a planned expansion to crossdomain ITC data (132 protein-ligand entries pending mapping).
- Kinetic head (k_on, k_off) currently produces zeros; planned for v_clean_006.

## Citation

If this work helps you, please cite:

```
Sword, A. & Hanya. (2026). Meridian × Gemma 4: Physics-informed
binding affinity as a Gemma 4 function-calling tool for neglected
disease drug discovery. Gemma 4 Good Hackathon submission.
https://github.com/q-sword/meridian-gemma-4-hackathon
```

## License

MIT — see [LICENSE](LICENSE). Note: model weights are *not* under MIT; contact Sword Labs for terms.

## Acknowledgments

Gemma 4 team at Google DeepMind for releasing a function-calling open-weights model.
BindingDB, ChEMBL, PDBbind, and KLIFS for the training data.
Medicines for Malaria Venture (MMV) for the open-source antimalarial pipeline that motivated this work.
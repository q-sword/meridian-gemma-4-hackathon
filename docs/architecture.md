# Architecture

## System overview

Four cooperating processes:

```
  USER  ───NL───▶  GEMMA 4 e4b  ──function_call─▶  MERIDIAN HTTP  ──inference─▶  MERIDIAN MODEL
                (via Ollama)                       SERVER (stdlib)             v_clean_005c (PyTorch)
   ▲                  │                                  │                          │
   │                  │                                  │                          │
   └──────clinical──────┘                                  ▼                          ▼
        answer with                              SmartCalibrator              physics_diagnostic
        uncertainty +                            (per-target ESM            (pKd_raw, dG_raw, dH_raw,
        thermo profile                             similarity)               mTdS_raw, components)
```

## Component responsibilities

### Gemma 4 e4b (Ollama)

- Parses natural-language queries ("how tight does X bind Y?")
- Selects and calls the appropriate tool from a small toolset
- Receives the rich JSON response and translates it into clinical voice
- Hard rules in system prompt: never guess numbers, always cite uncertainty, never give individualized medical advice
- Runs locally on consumer GPU (Gemma 4 e4b is 8B params, ~7 GB VRAM)

### Meridian HTTP server

- Wraps the Meridian inference library behind a minimal HTTP API
- Three endpoints:
  - `GET /health` — model identity and provenance
  - `GET /drugs` — curated drug panel (17 drugs)
  - `POST /predict_drug` — by drug name from curated panel
  - `POST /predict` — by arbitrary SMILES + UniProt
- Implementation: Python stdlib `http.server.ThreadingHTTPServer`. No Flask, no FastAPI, no extra runtime deps.

### Meridian model (v_clean_005c)

- 32.4M parameters, multi-modal:
  - **Molecule side:** SELFIES tokens → transformer encoder (6 layers, 8 heads, d=384, ffw=1536), plus 72-dim RDKit descriptors, plus 20-dim G2-projection features
  - **Protein side:** ESM-2 embeddings, AlphaFold pocket features, KLIFS kinase features (when applicable), per-residue features
  - **Joint head:** cross-attention fusion into a multi-task head producing pKd and physics quantities
- Multi-task losses during training: pKd (absolute kcal/mol), molecular physics features, pocket physics features, interaction component decomposition
- **Thermodynamic heads now bypassed at inference** — see [analytic_thermodynamics.md](analytic_thermodynamics.md)

### SmartCalibrator

Per-target post-hoc calibration. For each query target T:

1. **Direct anchor** (preferred): if the test set has measured binders for T, fit a linear calibration on those and apply.
2. **ESM-similarity extrapolation** (fallback): find the most similar anchored target T' by ESM-2 cosine similarity, transfer T''s calibration with explicit similarity score reported.
3. **No anchor** (rare): mark the prediction as `calibration_extrapolated=true` with widened CI.

This is what lets Gemma honestly report `calibration_source="direct_anchor_n1_ABL1"` vs `"esm_weighted_top=KIT_sim=0.911"`. Judges and clinicians see, per-prediction, whether the number is anchored or extrapolated.

## Data flow for a single query

```
user: "How tight does DSM265 bind PfDHODH? Is binding entropy- or enthalpy-driven?"
  │
  ▼
Gemma 4: receives query + tool schema
  │  (13 s reasoning, decides to call predict_drug_binding)
  ▼
Gemma 4 emits: predict_drug_binding(drug_name="DSM265", target_uniprot="Q08210")
  │
  ▼
Meridian server: POST /predict_drug with {drug_name, target_uniprot}
  │
  ├── lookup DSM265 SMILES from curated panel
  ├── lookup Q08210 (PfDHODH) protein index
  ├── featurize SMILES (SELFIES tokenize, RDKit descriptors, G2 projection)
  ├── retrieve target features (ESM-2, AlphaFold, per-residue)
  ├── run forward pass (28 ms on Blackwell)
  ├── apply SmartCalibrator on pKd_raw → pKd_calibrated, 80% CI
  ├── compute dG_kcal = -1.3643 * pKd_calibrated  (Van't Hoff exact)
  ├── compute dH_kcal from model dH head with linear calibration
  ├── compute mTdS_kcal = dG_kcal - dH_kcal  (Gibbs exact)
  ├── attach Lipinski/MW/logP/TPSA from RDKit
  ├── attach structural data flags (has_alphafold, has_klifs, has_per_residue)
  └── return JSON bundle (~30 fields)
  │
  ▼
Gemma 4: receives JSON
  │  (26 s composing the final answer)
  ▼
Gemma 4 writes: pKd 8.75 (CI 8.05–9.45) → 1.77 nM (CI 0.35–8.85 nM)
               dG −11.94 / dH −9.36 / mTdS −2.58 kcal/mol
               binding is enthalpy-driven (specific molecular interactions)
               calibration source: ESM-weighted from KIT (sim=0.911), extrapolated
               disclaimer: research context only

total wall time: ~40 seconds end-to-end
```

## Tool definitions (Gemma 4 system prompt)

Three function-calling tools (full schemas in `demo/gemma_meridian_demo.py`):

1. **`predict_drug_binding(drug_name, target_uniprot?)`** — curated panel lookup. UX-optimized for named drugs.
2. **`predict_smiles_binding(smiles, uniprot)`** — arbitrary SMILES for advanced users.
3. **`list_curated_drugs()`** — returns the 17-drug panel with primary targets.

The system prompt sets hard rules:
- Never guess pKd, Kd, IC50, ΔG, or ΔH. Always call the tool.
- Always cite 80% CI in both pKd and nM.
- Always cite calibration_source verbatim.
- Always interpret the thermodynamic decomposition.
- Never provide individualized medical advice.

This is the entirety of the "alignment" layer. The model's honesty comes from the calibrated tool, not from Gemma 4 being trustworthy on its own.

## Curated drug panel

17 drugs grouped by clinical context:

**Antimalarials (the Health & Sciences narrative):**
- DSM265 → PfDHODH (Q08210)
- Atovaquone → PfCytB (P28593)
- Pyrimethamine → PfDHFR (Q27738)

**Validated oncology / kinase:**
- Imatinib → BCR-ABL (P00519)

**Common drugs for calibration reference:**
- Aspirin, Ibuprofen, Caffeine, Methamphetamine, Penicillin_G, Trimethoprim, Testosterone, Celecoxib, Fluoxetine, Haloperidol, Atorvastatin, Enalapril, Alprenolol

Each drug has a primary target documented, but users can override via `target_uniprot` to probe off-target binding. (See demo case 2: Imatinib vs. SERT for off-target speculation.)

## Why stdlib HTTP server, not FastAPI

The goal is *reproducibility for judges*. A `pip install` step is one more thing to break. `http.server.ThreadingHTTPServer` is in every Python install since 3.0. We chose 80 lines of stdlib over 4 lines of framework code so that a judge cloning this repo can run it with zero dependency drift.
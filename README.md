# Readmission Risk

30-day hospital readmission prediction, built as a **production ML system** —
not just a model. FHIR-native feature engineering, a deployed FastAPI service,
drift monitoring, and a weekly champion/challenger retraining loop.

> **Honest framing:** everything here trains on [Synthea](https://github.com/synthetichealth/synthea)
> synthetic patients (~1,180 FHIR R4 bundles). No PHI, no data-use agreement, fully
> reproducible. The claim is not "this model would work in a hospital" — it is
> "here is a complete, operated ML system": parsing, features, deployment,
> monitoring, and retraining, all wired together.

## Architecture

```
Synthea FHIR bundles ──▶ src/features.py ──▶ feature store (parquet)
        │                        │  ▲ same code in training AND serving
        │                        │  │ (no train-serve skew)
        ▼                        ▼  │
src/train.py ──▶ models/model_<ver>.pkl ──▶ app/main.py (FastAPI + Docker)
   │                        │               (production serving layer)
   │                        ▼
   │                   ┌─────────────┐
   │                   │  weekly     │
   └──────────────────▶│  retrain    │
   frozen holdout ──▶  │  (champion/ │
   promotion gate ──▶  │  challenger)│──▶ space/ ──▶ Hugging Face (static)
                       └─────────────┘     ▲ in-browser ONNX demo:
   scripts/monitor.py (weekly PSI check) ──┘ no server, no data leaves
   data/metrics.json ◀── live status feed ──▶ the visitor's machine
```

Two serving surfaces, one model:

- **`app/` — FastAPI + Docker.** The production serving design: `/predict` accepts a
  feature dict or a raw FHIR bundle, `/metrics` exposes the prediction/latency
  counters the drift monitor polls.
- **`space/` — static demo on Hugging Face.** The champion is exported to ONNX and
  scored with ONNX Runtime Web directly in the visitor's browser. Pick a synthetic
  patient, see the live risk score, the factors behind it, and what actually happened.
  No backend, no data leaves the machine — which is why it runs on HF's free tier.

## Quickstart

```bash
# 1. data (official Synthea R4 sample, ~85 MB)
curl -L -o data/synthea_fhir_r4.zip \
  https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_fhir_r4_sep2019.zip
unzip -q data/synthea_fhir_r4.zip -d data/synthea

# 2. features (one row per inpatient discharge, 30-day readmission label)
python scripts/build_features.py

# 3. train (patient-grouped split; holdout frozen for all future comparisons)
python src/train.py --version v2026-10

# 4. serve
uvicorn app.main:app --port 8000
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"features": {"age": 72, "n_imp_12m": 3, ...}}'
# ...or post a raw FHIR bundle:
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d @patient_bundle_wrapped.json
```

## The update loop

- **Weekly retrain** (`scripts/weekly_run.py`, scheduled in CI): generates a
  fresh synthetic cohort (new seed = simulated arrivals), retrains a challenger
  on all non-holdout patients, and promotes it only if holdout AUC clears the
  champion minus a 0.01 gate. Skips when fewer than 75 new patients arrive.
  Every run writes `runs/manifest_<ver>.json`.
- **Weekly drift check** (`scripts/monitor.py`): polls the API's `/metrics`
  endpoint and computes PSI of the served prediction distribution against the
  training baseline. No retraining here — just a health signal.
- **Live status** (`data/metrics.json`): committed by CI after every run; the
  Mini Project Space write-up fetches it client-side, so the post always shows
  the current champion, training set size, holdout AUC, and last drift check
  with zero manual edits.

## Model (v2026-10)

| | holdout AUC | avg precision |
|---|---|---|
| Logistic regression (baseline) | 0.90 | 0.73 |
| LightGBM (champion) | 0.93 | 0.80 |

2,108 discharges / 283 patients; 33.5% 30-day readmission rate. Holdout is 57
patients, frozen for all future champion/challenger comparisons. Top features:
days since last discharge, medication count, prior inpatient count, eGFR,
sodium — utilization history dominates, as expected. The high AUC reflects
Synthea's strong patient-level autocorrelation; treat it as a systems demo,
not a clinical result.

## Deploy

- **Demo:** a static Hugging Face Space (`space/`), rebuilt by CI every month from the
  current champion. Set `HF_TOKEN` as a repo secret (fine-grained, write on the Space only).
- **Production design:** `Dockerfile` serves the FastAPI app (`app/main.py`). HF's Docker
  Spaces need a paid plan, so the live demo is the static ONNX version; the container
  remains the reference serving architecture in the repo.

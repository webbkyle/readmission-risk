"""30-day readmission risk API.

Endpoints:
  POST /predict  {"features": {...}} or {"fhir_bundle": {...}}
                 The FHIR path runs the SAME featurize_bundle() used in
                 training, so there is no train-serve skew.
  GET  /health   liveness + loaded model version
  GET  /metrics  request/latency/proba-distribution counters. The weekly
                 monitoring job polls this endpoint (the Space's disk is
                 ephemeral, so no log shipping is assumed).
"""
import json
import os
import threading
import time
from datetime import datetime, timezone

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import FEATURES, featurize_bundle

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model_v2026-10.pkl')
CARD_PATH = os.environ.get('CARD_PATH', 'models/model_v2026-10.card.json')

app = FastAPI(title='readmission-risk', version='0.1.0')

_model = None
_card = None
_features = FEATURES


def load_artifacts():
    global _model, _card, _features
    _model = joblib.load(MODEL_PATH)
    with open(CARD_PATH) as fh:
        _card = json.load(fh)
    _features = _card.get('features', FEATURES)


@app.on_event('startup')
def _startup():
    load_artifacts()


# ---------------- in-memory metrics (ephemeral-disk safe) ----------------
_lock = threading.Lock()
_stats = {
    'n_requests': 0,
    'n_errors': 0,
    'latency_ms_sum': 0.0,
    'latency_ms_max': 0.0,
    'proba_bins': [0] * 10,  # histogram of predicted probabilities
    'started_at': datetime.now(timezone.utc).isoformat(),
}


class PredictRequest(BaseModel):
    features: Optional[Dict[str, Any]] = Field(
        default=None, description='Feature dict keyed by feature name')
    fhir_bundle: Optional[Dict[str, Any]] = Field(
        default=None, description='Raw FHIR R4 Bundle; featurized server-side')


class PredictResponse(BaseModel):
    readmit_proba: float
    model_version: str
    latency_ms: float
    via: str  # 'features' or 'fhir_bundle'


def _row_from_features(d):
    return [d.get(f, float('nan')) for f in _features]


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    try:
        if req.fhir_bundle is not None:
            feat = featurize_bundle(req.fhir_bundle)
            if feat is None:
                raise HTTPException(400, 'bundle has no inpatient stay to score')
            row = [feat.get(f, float('nan')) for f in _features]
            via = 'fhir_bundle'
        elif req.features is not None:
            row = _row_from_features(req.features)
            via = 'features'
        else:
            raise HTTPException(400, 'provide "features" or "fhir_bundle"')
        proba = float(_model.predict_proba([row])[0][1])
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface as 500, count it
        with _lock:
            _stats['n_errors'] += 1
        raise HTTPException(500, 'prediction failed: %s' % exc)
    ms = (time.perf_counter() - t0) * 1000.0
    with _lock:
        _stats['n_requests'] += 1
        _stats['latency_ms_sum'] += ms
        _stats['latency_ms_max'] = max(_stats['latency_ms_max'], ms)
        _stats['proba_bins'][min(9, int(proba * 10))] += 1
    # structured log line: the monitoring job tails these in real deployments
    print(json.dumps({'ts': datetime.now(timezone.utc).isoformat(),
                      'model_version': _card['version'], 'via': via,
                      'readmit_proba': round(proba, 4),
                      'latency_ms': round(ms, 1)}), flush=True)
    return PredictResponse(readmit_proba=proba, model_version=_card['version'],
                           latency_ms=ms, via=via)


@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': _card['version'] if _card else None,
            'n_features': len(_features)}


@app.get('/metrics')
def metrics():
    with _lock:
        s = dict(_stats)
    n = s['n_requests']
    return {
        'model_version': _card['version'] if _card else None,
        'n_requests': n,
        'n_errors': s['n_errors'],
        'latency_ms_mean': (s['latency_ms_sum'] / n) if n else 0.0,
        'latency_ms_max': s['latency_ms_max'],
        'proba_bins': s['proba_bins'],
        'started_at': s['started_at'],
    }


@app.get('/features')
def feature_list():
    """Ordered feature names the /predict features dict should use."""
    return {'features': _features}

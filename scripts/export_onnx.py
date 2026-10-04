#!/usr/bin/env python3
"""Export the champion LightGBM model to ONNX for in-browser inference.

The Space is static, so scoring happens client-side with onnxruntime-web.
Median imputation happens in JS (medians.json); the ONNX graph is the pure
LightGBM classifier. Verifies parity against the original model on the
frozen holdout before writing artifacts.

Usage: python scripts/export_onnx.py --version v2026-10 --out space/
Writes: space/model.onnx, space/medians.json, space/features.json
"""
import argparse, json, os

import joblib
import numpy as np
import pandas as pd
from onnxmltools import convert_lightgbm
from onnxmltools.utils import save_model
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort

ROOT = os.path.join(os.path.dirname(__file__), '..')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--out', default=os.path.join(ROOT, 'space'))
    args = ap.parse_args()

    card = json.load(open(os.path.join(ROOT, 'models', 'model_%s.card.json' % args.version)))
    feats = card['features']
    model = joblib.load(os.path.join(ROOT, 'models', 'model_%s.pkl' % args.version))

    df = pd.read_parquet(os.path.join(ROOT, 'data', 'features.parquet'))
    ho = np.load(os.path.join(ROOT, 'models', 'holdout_idx.npy'))
    medians = df[feats].median().to_dict()

    initial_types = [('input', FloatTensorType([None, len(feats)]))]
    onx = convert_lightgbm(model, initial_types=initial_types,
                           target_opset=15, zipmap=False)

    os.makedirs(args.out, exist_ok=True)
    save_model(onx, os.path.join(args.out, 'model.onnx'))
    json.dump({'features': feats, 'medians': medians, 'model_version': args.version},
              open(os.path.join(args.out, 'model_meta.json'), 'w'))

    # parity check on frozen holdout
    Xho = df[feats].values[ho]
    med = np.array([medians[f] for f in feats])
    Xho_imp = np.where(np.isnan(Xho), med, Xho).astype(np.float32)
    sess = ort.InferenceSession(os.path.join(args.out, 'model.onnx'))
    p_onnx = np.asarray(sess.run(None, {'input': Xho_imp})[1])[:, 1]
    p_orig = model.predict_proba(Xho_imp)[:, 1]
    maxdiff = float(np.max(np.abs(p_onnx - p_orig)))
    print('holdout n=%d, max |p_onnx - p_orig| = %.2e' % (len(Xho), maxdiff))
    assert maxdiff < 1e-5, 'ONNX parity failed'
    print('wrote %s/model.onnx + model_meta.json' % args.out)


if __name__ == '__main__':
    main()

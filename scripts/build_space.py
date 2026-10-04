#!/usr/bin/env python3
"""Build the static Space's data artifacts.

Generates (into space/):
  model.onnx, model_meta.json   via export_onnx.py
  patients.json                 one record per discharge (2,108 labeled rows)
  feature_info.json              display names + LightGBM gain importances
  metrics.json                   copy of data/metrics.json (fallback; the page
                                 prefers the live file from GitHub)

Usage: python scripts/build_space.py [--version v2026-10] [--data-only]
"""
import argparse, json, os

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..')
SPACE = os.path.join(ROOT, 'space')

DISPLAY = {
    'age': 'Age', 'sex_male': 'Male', 'los_days': 'Length of stay (days)',
    'n_imp_12m': 'Inpatient stays, prior 12 mo', 'n_ed_12m': 'ED visits, prior 12 mo',
    'n_amb_12m': 'Outpatient visits, prior 12 mo', 'n_imp_all': 'Lifetime inpatient stays',
    'days_since_last_discharge': 'Days since last discharge',
    'n_cond_active': 'Active conditions', 'n_meds': 'Medications',
    'n_proc_12m': 'Procedures, prior 12 mo',
    'flag_diabetes': 'Diabetes', 'flag_hypertension': 'Hypertension',
    'flag_heart_failure': 'Heart failure', 'flag_copd': 'COPD',
    'flag_ckd': 'Chronic kidney disease', 'flag_depression': 'Depression',
    'flag_obesity': 'Obesity',
    'bmi': 'BMI', 'sys_bp': 'Systolic BP', 'dia_bp': 'Diastolic BP',
    'hba1c': 'HbA1c (%)', 'glucose': 'Glucose (mg/dL)',
    'creatinine': 'Creatinine (mg/dL)', 'egfr': 'eGFR',
    'sodium': 'Sodium (mEq/L)', 'potassium': 'Potassium (mEq/L)',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='v2026-10')
    ap.add_argument('--data-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(SPACE, exist_ok=True)

    if not args.data_only:
        import subprocess, sys
        subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'export_onnx.py'),
                        '--version', args.version, '--out', SPACE], check=True)

    meta = json.load(open(os.path.join(SPACE, 'model_meta.json')))
    feats = meta['features']
    model = joblib.load(os.path.join(ROOT, 'models', 'model_%s.pkl' % args.version))
    gains = model.booster_.feature_importance(importance_type='gain')
    order = np.argsort(gains)[::-1]
    df = pd.read_parquet(os.path.join(ROOT, 'data', 'features.parquet'))
    display_medians = df[feats].median().to_dict()
    json.dump({
        'display': {f: DISPLAY.get(f, f) for f in feats},
        'importance': [{'feature': feats[i], 'gain': float(gains[i])} for i in order],
        'medians': display_medians,
    }, open(os.path.join(SPACE, 'feature_info.json'), 'w'))

    df = df.sort_values('discharge_date')
    pmap = {pid: 'P%03d' % (i + 1) for i, pid in enumerate(df['patient_id'].unique())}
    patients = []
    for k, r in enumerate(df.itertuples()):
        d = r._asdict()
        patients.append({
            'id': 'D%04d' % (k + 1),
            'plabel': '%s · %s' % (pmap[d['patient_id']], str(d['discharge_date'])[:10]),
            'age': int(r.age), 'male': int(r.sex_male),
            'discharge_date': str(r.discharge_date),
            'features': [None if pd.isna(d[f]) else round(float(d[f]), 3)
                         for f in feats],
            'readmitted': int(r.readmit_30d),
        })
    json.dump(patients, open(os.path.join(SPACE, 'patients.json'), 'w'))
    print('discharges: %d' % len(patients))

    status = json.load(open(os.path.join(ROOT, 'data', 'metrics.json')))
    json.dump(status, open(os.path.join(SPACE, 'metrics.json'), 'w'))
    print('wrote space data artifacts')


if __name__ == '__main__':
    main()

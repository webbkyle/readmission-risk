#!/usr/bin/env python3
"""Build the static Space's data artifacts.

Generates (into space/):
  model.onnx, model_meta.json   via export_onnx.py
  patients.json                 one record per patient (most recent discharge)
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
    json.dump({
        'display': {f: DISPLAY.get(f, f) for f in feats},
        'importance': [{'feature': feats[i], 'gain': float(gains[i])} for i in order],
        'medians': meta['medians'],
    }, open(os.path.join(SPACE, 'feature_info.json'), 'w'))

    df = pd.read_parquet(os.path.join(ROOT, 'data', 'features.parquet'))
    df = df.sort_values('discharge_date').groupby('patient_id').tail(1)
    patients = []
    for k, r in enumerate(df.itertuples()):
        patients.append({
            'id': 'P%03d' % (k + 1),
            'age': int(r.age), 'male': int(r.sex_male),
            'discharge_date': str(r.discharge_date),
            'features': [None if pd.isna(r._asdict()[f]) else round(float(r._asdict()[f]), 3)
                         for f in feats],
            'readmitted': int(r.readmit_30d),
        })
    json.dump(patients, open(os.path.join(SPACE, 'patients.json'), 'w'))
    print('patients: %d' % len(patients))

    status = json.load(open(os.path.join(ROOT, 'data', 'metrics.json')))
    json.dump(status, open(os.path.join(SPACE, 'metrics.json'), 'w'))
    print('wrote space data artifacts')


if __name__ == '__main__':
    main()

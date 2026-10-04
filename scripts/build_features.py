#!/usr/bin/env python3
"""Build the labeled feature table from Synthea FHIR bundles.

Usage: python scripts/build_features.py [--data DIR] [--out FILE]
Writes one row per inpatient discharge with a 30-day readmission label.
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
from features import discharge_rows, FEATURES, LABEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/synthea/fhir')
    ap.add_argument('--out', default='data/features.parquet')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, '*.json')))
    print('bundles: %d' % len(files))
    rows = []
    for i, f in enumerate(files):
        with open(f) as fh:
            bundle = json.load(fh)
        pid = os.path.basename(f)
        rows.extend(discharge_rows(bundle, pid))
        if (i + 1) % 200 == 0:
            print('  %d/%d bundles, %d rows' % (i + 1, len(files), len(rows)))
    df = pd.DataFrame(rows)
    cols = ['discharge_id', 'patient_id', 'discharge_date'] + FEATURES + [LABEL]
    df = df[cols]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_parquet(args.out, index=False)
    print('wrote %s: %s rows x %s cols' % (args.out, len(df), len(df.columns)))
    print('positive rate: %.3f' % df[LABEL].mean())
    print('patients: %d' % df['patient_id'].nunique())


if __name__ == '__main__':
    main()

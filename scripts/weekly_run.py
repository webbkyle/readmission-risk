#!/usr/bin/env python3
"""Weekly champion/challenger retraining loop.

1. Builds features for a NEW cohort of FHIR bundles (--new-bundles).
2. Skips retraining if fewer than --min-new-patients new patients arrived.
3. Appends new rows to the feature store (deduped by discharge_id).
4. Trains a challenger on all non-holdout patients (frozen holdout kept).
5. Promotes the challenger only if holdout AUC >= champion AUC - --gate.
6. Writes runs/manifest_<version>.json and updates data/metrics.json
   (the feed behind the write-up's live status panel).

In CI, the new cohort is generated with Synthea (see
.github/workflows/weekly-retrain.yml). Pushing the promoted artifact to
Hugging Face Spaces happens in CI via the HF API (needs HF_TOKEN secret).

Usage:
  python scripts/weekly_run.py --new-bundles data/synthea-new/fhir \
      --version v2026-10-05-r37224725536 [--gate 0.01] [--min-new-patients 75] [--dry-run]

Version strings must be unique per run: CI appends the GitHub run number
(vYYYY-MM-DD-rNNNN) so a same-day re-run never collides with an earlier run.
"""
import argparse, glob, json, os, subprocess, sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import discharge_rows, FEATURES, LABEL

ROOT = os.path.join(os.path.dirname(__file__), '..')


def load_status():
    with open(os.path.join(ROOT, 'data', 'metrics.json')) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new-bundles', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--gate', type=float, default=0.01)
    ap.add_argument('--min-new-patients', type=int, default=100)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    status = load_status()
    champion_version = status['champion_version']
    print('champion: %s' % champion_version)

    # fail fast on version collision: never silently overwrite a prior run
    for p in [os.path.join(ROOT, 'models', 'model_%s.pkl' % args.version),
              os.path.join(ROOT, 'runs', 'manifest_%s.json' % args.version)]:
        if os.path.exists(p):
            raise SystemExit('version collision: %s already exists -- use a unique --version' % p)

    # ---- 1-2. featurize new cohort, check volume gate ----
    files = sorted(glob.glob(os.path.join(args.new_bundles, '*.json')))
    print('new bundles: %d' % len(files))
    rows = []
    for f in files:
        with open(f) as fh:
            bundle = json.load(fh)
        rows.extend(discharge_rows(bundle, os.path.basename(f)))
    new_df = pd.DataFrame(rows)
    n_new_patients = new_df['patient_id'].nunique() if len(new_df) else 0
    print('new patients: %d, new discharges: %d' % (n_new_patients, len(new_df)))
    if n_new_patients < args.min_new_patients:
        msg = 'skipped: only %d new patients (< %d minimum)' % (
            n_new_patients, args.min_new_patients)
        print(msg)
        status['last_run'] = str(date.today())
        status['last_outcome'] = msg
        status['history'].append({'date': str(date.today()), 'event': 'skipped',
                                  'reason': msg})
        if not args.dry_run:
            json.dump(status, open(os.path.join(ROOT, 'data', 'metrics.json'), 'w'), indent=2)
        return

    # ---- 3. append to feature store ----
    store_path = os.path.join(ROOT, 'data', 'features.parquet')
    store = pd.read_parquet(store_path)
    cols = ['discharge_id', 'patient_id', 'discharge_date'] + FEATURES + [LABEL]
    new_df = new_df[cols]
    store = pd.concat([store, new_df]).drop_duplicates('discharge_id').reset_index(drop=True)
    print('feature store: %d rows / %d patients' % (len(store), store['patient_id'].nunique()))
    if not args.dry_run:
        store.to_parquet(store_path, index=False)

    # ---- 4. train challenger (frozen holdout) ----
    cmd = [sys.executable, os.path.join(ROOT, 'src', 'train.py'),
           '--features', store_path,
           '--out', os.path.join(ROOT, 'models'),
           '--version', args.version,
           '--holdout-patients', os.path.join(ROOT, 'models', 'holdout_patients.json')]
    print('training challenger:', ' '.join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)

    # ---- 5. promotion gate on the frozen holdout ----
    champ_card = json.load(open(os.path.join(ROOT, 'models', 'model_%s.card.json' % champion_version)))
    chal_card = json.load(open(os.path.join(ROOT, 'models', 'model_%s.card.json' % args.version)))
    champ_auc = champ_card['metrics_holdout']['lightgbm']['auc']
    chal_auc = chal_card['metrics_holdout']['lightgbm']['auc']
    promoted = chal_auc >= champ_auc - args.gate
    outcome = ('PROMOTED %s (holdout AUC %.4f >= %.4f - gate %.3f)'
               % (args.version, chal_auc, champ_auc, args.gate) if promoted else
               'no promotion: challenger %s AUC %.4f < gate %.4f (champion %.4f)'
               % (args.version, chal_auc, champ_auc - args.gate, champ_auc))
    print(outcome)

    # ---- 6. manifest + status feed ----
    manifest = {
        'date': str(date.today()),
        'challenger_version': args.version,
        'champion_version': champion_version,
        'challenger_holdout_auc': chal_auc,
        'champion_holdout_auc': champ_auc,
        'gate': args.gate,
        'promoted': promoted,
        'n_new_patients': int(n_new_patients),
        'n_train_rows': chal_card['n_train_rows'],
    }
    os.makedirs(os.path.join(ROOT, 'runs'), exist_ok=True)
    if not args.dry_run:
        json.dump(manifest, open(os.path.join(ROOT, 'runs', 'manifest_%s.json' % args.version), 'w'), indent=2)
        if promoted:
            status['champion_version'] = args.version
            status['trained_on'] = chal_card['trained_on']
            status['n_train_patients'] = chal_card['n_train_patients']
            status['n_train_rows'] = chal_card['n_train_rows']
            status['holdout_auc'] = round(chal_auc, 4)
            status['holdout_avg_precision'] = round(
                chal_card['metrics_holdout']['lightgbm']['avg_precision'], 4)
        status['last_run'] = str(date.today())
        status['last_outcome'] = outcome
        status['history'].append({'date': str(date.today()),
                                  'event': 'promoted' if promoted else 'no promotion',
                                  'version': args.version,
                                  'holdout_auc': round(chal_auc, 4)})
        json.dump(status, open(os.path.join(ROOT, 'data', 'metrics.json'), 'w'), indent=2)
        print('metrics.json updated')
    else:
        print('[dry run] manifest:', json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

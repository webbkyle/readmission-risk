#!/usr/bin/env python3
"""Train 30-day readmission models.

- Splits BY PATIENT (GroupShuffleSplit): a frozen holdout is kept for all
  future champion/challenger comparisons.
- Baseline: logistic regression (median-imputed). Challenger: LightGBM
  (native NaN handling).
- Saves model artifact + model card JSON (version, metrics, feature list).

Usage: python src/train.py [--features data/features.parquet] [--out models/]
                           [--version v2026-10] [--seed 42]
"""
import argparse, hashlib, json, os, sys
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES, LABEL

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def metrics(y, p):
    return {
        'auc': float(roc_auc_score(y, p)),
        'avg_precision': float(average_precision_score(y, p)),
        'brier': float(brier_score_loss(y, p)),
        'n': int(len(y)),
        'positive_rate': float(np.mean(y)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='data/features.parquet')
    ap.add_argument('--out', default='models')
    ap.add_argument('--version', default='v2026-10')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--holdout-patients', default=None,
                    help='JSON list of patient_ids to freeze as holdout (champion/challenger comparability)')
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    # drop zero-variance columns (e.g. flags absent from this synthetic cohort)
    feats = [c for c in FEATURES if df[c].nunique(dropna=False) > 1]
    dropped = [c for c in FEATURES if c not in feats]
    print('features: %d kept, dropped zero-variance: %s' % (len(feats), dropped))

    X = df[feats].values
    y = df[LABEL].values
    groups = df['patient_id'].values

    # frozen holdout, split by patient
    if args.holdout_patients:
        ho_pids = set(json.load(open(args.holdout_patients)))
        ho_idx = np.where(np.isin(groups, list(ho_pids)))[0]
        tr_idx = np.where(~np.isin(groups, list(ho_pids)))[0]
        print('using frozen holdout: %d patients' % len(ho_pids))
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        tr_idx, ho_idx = next(gss.split(X, y, groups))
    Xtr, Xho, ytr, yho = X[tr_idx], X[ho_idx], y[tr_idx], y[ho_idx]
    print('train: %d rows / %d patients | holdout: %d rows / %d patients' % (
        len(Xtr), len(np.unique(groups[tr_idx])), len(Xho), len(np.unique(groups[ho_idx]))))
    print('train pos rate: %.3f | holdout pos rate: %.3f' % (ytr.mean(), yho.mean()))

    results = {}

    # ---- baseline: logistic regression ----
    lr = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                       LogisticRegression(max_iter=2000, C=1.0))
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xho)[:, 1]
    results['logistic_regression'] = metrics(yho, p_lr)
    print('LR holdout:', {k: round(v, 4) for k, v in results['logistic_regression'].items()})

    # ---- LightGBM with 5-fold grouped CV on train ----
    if HAS_LGBM:
        gkf = GroupKFold(n_splits=5)
        cv_aucs = []
        for ti, vi in gkf.split(Xtr, ytr, groups[tr_idx]):
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   num_leaves=31, min_child_samples=20,
                                   random_state=args.seed, verbose=-1)
            m.fit(Xtr[ti], ytr[ti])
            cv_aucs.append(roc_auc_score(ytr[vi], m.predict_proba(Xtr[vi])[:, 1]))
        print('LGBM 5-fold grouped CV AUC: %.4f +- %.4f' % (np.mean(cv_aucs), np.std(cv_aucs)))
        results['lightgbm_cv_auc_mean'] = float(np.mean(cv_aucs))
        results['lightgbm_cv_auc_std'] = float(np.std(cv_aucs))

        final = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   num_leaves=31, min_child_samples=20,
                                   random_state=args.seed, verbose=-1)
        final.fit(Xtr, ytr)
        p_lgbm = final.predict_proba(Xho)[:, 1]
        results['lightgbm'] = metrics(yho, p_lgbm)
        print('LGBM holdout:', {k: round(v, 4) for k, v in results['lightgbm'].items()})
        imp = sorted(zip(feats, final.feature_importances_), key=lambda x: -x[1])[:10]
        print('top features:', [(f, int(v)) for f, v in imp])
        results['top_features'] = [f for f, _ in imp]
        champion, champion_name = final, 'lightgbm'
    else:
        champion, champion_name = lr, 'logistic_regression'

    # ---- save artifact + model card ----
    os.makedirs(args.out, exist_ok=True)
    with open(args.features, 'rb') as fh:
        data_hash = hashlib.sha256(fh.read()).hexdigest()[:12]
    joblib.dump(champion, os.path.join(args.out, 'model_%s.pkl' % args.version))
    card = {
        'version': args.version,
        'model': champion_name,
        'trained_on': str(date.today()),
        'features': feats,
        'dropped_zero_variance': dropped,
        'n_train_rows': int(len(Xtr)),
        'n_train_patients': int(len(np.unique(groups[tr_idx]))),
        'n_holdout_rows': int(len(Xho)),
        'n_holdout_patients': int(len(np.unique(groups[ho_idx]))),
        'train_positive_rate': float(ytr.mean()),
        'data_sha': data_hash,
        'seed': args.seed,
        'metrics_holdout': results,
    }
    with open(os.path.join(args.out, 'model_%s.card.json' % args.version), 'w') as fh:
        json.dump(card, fh, indent=2)
    # frozen holdout indices for all future champion/challenger comparisons
    np.save(os.path.join(args.out, 'holdout_idx.npy'), ho_idx)
    print('saved models/%s.pkl + card; holdout frozen (%d rows)' % (args.version, len(ho_idx)))


if __name__ == '__main__':
    main()

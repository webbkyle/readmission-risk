#!/usr/bin/env python3
"""Weekly drift check.

Polls the deployed API's /metrics endpoint (its proba histogram), computes
the Population Stability Index (PSI) against the training baseline, and
updates data/metrics.json's drift section -- the feed behind the write-up's
live status panel. No retraining here; that is scripts/monthly_run.py.

PSI guide: < 0.1 no significant change, 0.1-0.25 small change, > 0.25 big shift.

Usage: python scripts/monitor.py [--api URL] [--psi-warn 0.25]
"""
import argparse, json, math, os, urllib.request
from datetime import date

ROOT = os.path.join(os.path.dirname(__file__), '..')


def psi(expected, actual, eps=1e-6):
    total = 0.0
    for e, a in zip(expected, actual):
        e = max(e, eps)
        a = max(a, eps)
        total += (a - e) * math.log(a / e)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default='http://localhost:8000')
    ap.add_argument('--psi-warn', type=float, default=0.25)
    ap.add_argument('--min-requests', type=int, default=50,
                    help='skip PSI when fewer requests than this were served')
    args = ap.parse_args()

    baseline = json.load(open(os.path.join(ROOT, 'data', 'baseline.json')))
    with urllib.request.urlopen(args.api.rstrip('/') + '/metrics', timeout=30) as r:
        m = json.load(r)

    n = m['n_requests']
    status = json.load(open(os.path.join(ROOT, 'data', 'metrics.json')))
    if n < args.min_requests:
        result = {'status': 'insufficient traffic',
                  'n_requests': n, 'max_psi': None,
                  'date': str(date.today())}
        print('only %d requests served; skipping PSI' % n)
    else:
        e = baseline['proba_bin_counts']
        a = m['proba_bins']
        e = [x / sum(e) for x in e]
        a = [x / sum(a) for x in a]
        v = psi(e, a)
        result = {'status': 'WARNING' if v >= args.psi_warn else 'OK',
                  'n_requests': n, 'max_psi': round(v, 4),
                  'psi_threshold': args.psi_warn,
                  'latency_ms_mean': round(m['latency_ms_mean'], 1),
                  'latency_ms_max': round(m['latency_ms_max'], 1),
                  'date': str(date.today())}
        print('PSI=%.4f -> %s' % (v, result['status']))
    status['drift'] = result
    json.dump(status, open(os.path.join(ROOT, 'data', 'metrics.json'), 'w'), indent=2)
    print('metrics.json drift section updated')


if __name__ == '__main__':
    main()

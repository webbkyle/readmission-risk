"""FHIR-native feature engineering for 30-day readmission prediction.

The SAME functions are used in training (scripts/build_features.py) and in
serving (app/main.py /predict accepts a raw FHIR bundle), so there is no
train-serve skew: the serving path literally calls featurize_bundle().
"""
from datetime import datetime, timedelta

FEATURES = [
    'age', 'sex_male',
    'los_days',
    'n_imp_12m', 'n_ed_12m', 'n_amb_12m',
    'n_imp_all', 'days_since_last_discharge',
    'n_conditions', 'n_meds', 'n_proc_12m',
    'flag_diabetes', 'flag_hypertension', 'flag_heart_failure',
    'flag_copd', 'flag_ckd', 'flag_depression', 'flag_obesity',
    'bmi', 'sbp', 'dbp', 'hba1c', 'glucose', 'creatinine', 'egfr',
    'sodium', 'potassium',
]

LABEL = 'readmit_30d'

# LOINC codes for the vitals/labs we keep (last value before discharge)
VITALS = {
    'bmi': '39156-5',
    'sbp': '8480-6',
    'dbp': '8462-4',
    'hba1c': '4548-4',
    'glucose': '2339-0',
    'creatinine': '38483-4',
    'egfr': '33914-3',
    'sodium': '2947-0',
    'potassium': '6298-4',
}

# substring match on lowercased condition display text
CHRONIC = {
    'flag_diabetes': ['diabetes'],
    'flag_hypertension': ['hypertension'],
    'flag_heart_failure': ['heart failure'],
    'flag_copd': ['copd', 'chronic obstructive pulmonary'],
    'flag_ckd': ['chronic kidney disease'],
    'flag_depression': ['depression'],
    'flag_obesity': ['obesity'],
}
CHRONIC_EXCLUDE = {'flag_diabetes': ['prediabetes']}


def _dt(s):
    return datetime.fromisoformat(s) if s else None


def _res_time(r):
    """Best-effort event datetime for a FHIR resource."""
    rt = r['resourceType']
    if rt == 'Encounter':
        return _dt(r.get('period', {}).get('start'))
    if rt == 'Observation':
        return _dt(r.get('effectiveDateTime'))
    if rt == 'Condition':
        return _dt(r.get('onsetDateTime'))
    if rt == 'MedicationRequest':
        return _dt(r.get('authoredOn'))
    if rt == 'Procedure':
        return _dt(r.get('performedDateTime') or (r.get('performedPeriod') or {}).get('start'))
    if rt == 'Immunization':
        return _dt(r.get('occurrenceDateTime'))
    return None


def _obs_values(obs):
    """Yield (loinc_code, value) for an Observation, including panel components."""
    out = []
    for c in obs.get('code', {}).get('coding', []):
        code = c.get('code')
        v = (obs.get('valueQuantity') or {}).get('value')
        if code and v is not None:
            out.append((code, float(v)))
    for comp in obs.get('component', []):
        for c in comp.get('code', {}).get('coding', []):
            code = c.get('code')
            v = (comp.get('valueQuantity') or {}).get('value')
            if code and v is not None:
                out.append((code, float(v)))
    return out


def _display(r):
    parts = []
    for c in r.get('code', {}).get('coding', []):
        if c.get('display'):
            parts.append(c['display'])
    if r.get('code', {}).get('text'):
        parts.append(r['code']['text'])
    return ' '.join(parts).lower()


def parse_bundle(bundle):
    """Split a Synthea patient Bundle into demographics + timestamped resources."""
    patient = None
    enc, cond, obs, med, proc = [], [], [], [], []
    for e in bundle.get('entry', []):
        r = e['resource']
        rt = r['resourceType']
        if rt == 'Patient':
            patient = r
        elif rt == 'Encounter':
            t = _res_time(r)
            if t:
                enc.append((t, r))
        elif rt == 'Condition':
            cond.append(r)
        elif rt == 'Observation':
            t = _res_time(r)
            if t:
                obs.append((t, r))
        elif rt == 'MedicationRequest':
            med.append(r)
        elif rt == 'Procedure':
            t = _res_time(r)
            if t:
                proc.append((t, r))
    if patient is None:
        return None
    enc.sort(key=lambda x: x[0])
    obs.sort(key=lambda x: x[0])
    proc.sort(key=lambda x: x[0])
    return {'patient': patient, 'enc': enc, 'cond': cond, 'obs': obs,
            'med': med, 'proc': proc}


def _inpatient_stays(parsed):
    stays = []
    for t, r in parsed['enc']:
        if r.get('class', {}).get('code') == 'IMP' and r.get('status') == 'finished':
            per = r.get('period', {})
            s, e = _dt(per.get('start')), _dt(per.get('end'))
            if s and e:
                stays.append((s, e))
    stays.sort()
    return stays


def featurize_at(parsed, adm_start, dis_end):
    """Feature dict for one inpatient stay. Uses only data known at discharge."""
    p = parsed['patient']
    birth = _dt(p.get('birthDate'))
    age = (dis_end.date() - birth.date()).days / 365.25 if birth else float('nan')
    sex_male = 1 if (p.get('gender') == 'male') else 0
    los_days = (dis_end - adm_start).total_seconds() / 86400.0
    # Synthea occasionally emits artifact stays lasting years; winsorize.
    los_days = min(los_days, 60.0)

    yr = timedelta(days=365)
    n_imp_12m = n_ed_12m = n_amb_12m = n_imp_all = 0
    prev_dis = None
    for t, r in parsed['enc']:
        if t >= adm_start:
            break
        cls = r.get('class', {}).get('code')
        per = r.get('period', {})
        e = _dt(per.get('end'))
        if cls == 'IMP' and r.get('status') == 'finished' and e:
            n_imp_all += 1
            if t >= adm_start - yr:
                n_imp_12m += 1
            # only ended-before-admission stays count as "previous discharge";
            # an overlapping stay means days-since-discharge is 0
            if e <= adm_start and (prev_dis is None or e > prev_dis):
                prev_dis = e
        elif cls == 'EMER' and t >= adm_start - yr:
            n_ed_12m += 1
        elif cls == 'AMB' and t >= adm_start - yr:
            n_amb_12m += 1
    if prev_dis is not None:
        days_since = (adm_start - prev_dis).total_seconds() / 86400.0
    elif n_imp_all > 0:
        days_since = 0.0  # overlapping/transfer stay
    else:
        days_since = float('nan')

    n_conditions = 0
    flags = {k: 0 for k in CHRONIC}
    for r in parsed['cond']:
        onset = _dt(r.get('onsetDateTime'))
        if onset and onset > dis_end:
            continue
        abate = _dt(r.get('abatementDateTime'))
        if abate and abate <= dis_end:
            continue
        n_conditions += 1
        txt = _display(r)
        for fk, keys in CHRONIC.items():
            if any(k in txt for k in keys) and not any(x in txt for x in CHRONIC_EXCLUDE.get(fk, [])):
                flags[fk] = 1

    n_meds = sum(1 for r in parsed['med']
                 if (_dt(r.get('authoredOn')) or dis_end) <= dis_end)
    n_proc_12m = sum(1 for t, r in parsed['proc'] if t <= dis_end and t >= adm_start - yr)

    last = {}
    for t, r in parsed['obs']:
        if t > dis_end:
            break
        for code, v in _obs_values(r):
            last[code] = (t, v)  # keep latest; list is time-sorted
    vit = {}
    for name, code in VITALS.items():
        vit[name] = last[code][1] if code in last else float('nan')

    feat = {
        'age': age, 'sex_male': sex_male, 'los_days': los_days,
        'n_imp_12m': n_imp_12m, 'n_ed_12m': n_ed_12m, 'n_amb_12m': n_amb_12m,
        'n_imp_all': n_imp_all, 'days_since_last_discharge': days_since,
        'n_conditions': n_conditions, 'n_meds': n_meds, 'n_proc_12m': n_proc_12m,
        **flags, **vit,
    }
    return {k: feat[k] for k in FEATURES}


def discharge_rows(bundle, pid):
    """One labeled row per inpatient discharge. Returns list of dicts."""
    parsed = parse_bundle(bundle)
    if parsed is None:
        return []
    stays = _inpatient_stays(parsed)
    adm_starts = sorted(s for s, _ in stays)
    rows = []
    for i, (s, e) in enumerate(stays):
        label = 0
        for a in adm_starts:
            if timedelta(0) < (a - e) <= timedelta(days=30):
                label = 1
                break
        feat = featurize_at(parsed, s, e)
        feat.update({'patient_id': pid,
                     'discharge_id': '%s#%d' % (pid, i),
                     'discharge_date': e.isoformat(),
                     LABEL: label})
        rows.append(feat)
    return rows


def featurize_bundle(bundle):
    """Serving path: features for the patient's most recent inpatient discharge.

    Used by POST /predict when given a raw FHIR bundle. Same code as training.
    """
    parsed = parse_bundle(bundle)
    if parsed is None:
        return None
    stays = _inpatient_stays(parsed)
    if not stays:
        return None
    s, e = stays[-1]
    return featurize_at(parsed, s, e)

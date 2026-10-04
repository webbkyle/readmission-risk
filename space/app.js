/* Readmission-risk static demo: scoring runs in-browser via onnxruntime-web. */
let META, FINFO, PATIENTS, session;
const COUNT = new Set(['age','los_days','n_imp_12m','n_ed_12m','n_amb_12m','n_imp_all',
  'days_since_last_discharge','n_cond_active','n_meds','n_proc_12m']);
const FLAGS = new Set(['flag_diabetes','flag_hypertension','flag_heart_failure','flag_copd',
  'flag_ckd','flag_depression','flag_obesity']);

function fmt(f, v) {
  if (v == null) return 'n/a';
  if (FLAGS.has(f)) return v >= 0.5 ? 'yes' : 'no';
  if (COUNT.has(f)) return String(Math.round(v));
  return String(Math.round(v * 10) / 10);
}

async function init() {
  const [meta, finfo, patients] = await Promise.all([
    fetch('model_meta.json').then(r => r.json()),
    fetch('feature_info.json').then(r => r.json()),
    fetch('patients.json').then(r => r.json()),
  ]);
  META = meta; FINFO = finfo; PATIENTS = patients;
  session = await ort.InferenceSession.create('model.onnx');
  document.getElementById('search').addEventListener('input', e => renderList(e.target.value));
  document.getElementById('random').addEventListener('click', () =>
    select(PATIENTS[Math.floor(Math.random() * PATIENTS.length)].id));
  renderList('');
  loadStatus();
  renderModelCard();
  select(PATIENTS[Math.floor(Math.random() * PATIENTS.length)].id);
}

function renderList(q) {
  q = q.trim().toLowerCase();
  const box = document.getElementById('plist');
  box.innerHTML = '';
  const hits = PATIENTS.filter(p =>
    !q || p.id.toLowerCase().includes(q) || String(p.age) === q);
  for (const p of hits.slice(0, 400)) {
    const b = document.createElement('button');
    b.textContent = `${p.id} · age ${p.age} · ${p.male ? 'M' : 'F'}`;
    b.dataset.pid = p.id;
    b.onclick = () => select(p.id);
    box.appendChild(b);
  }
}

/* missing values pass through as NaN -- the model was trained with LightGBM's
   native NaN handling, and the ONNX graph replicates it exactly */
function toInput(feats) {
  return feats.map(v => v == null ? NaN : v);
}

async function score(p) {
  const x = new Float32Array(toInput(p.features));
  const t = new ort.Tensor('float32', x, [1, META.features.length]);
  const out = await session.run({ input: t });
  return out.probabilities.data[1];
}

function band(p) {
  if (p < 0.15) return ['low', 'Low'];
  if (p < 0.40) return ['mod', 'Moderate'];
  return ['high', 'High'];
}

async function select(pid) {
  const p = PATIENTS.find(x => x.id === pid);
  document.querySelectorAll('#plist button').forEach(b =>
    b.classList.toggle('active', b.dataset.pid === pid));
  document.getElementById('pname').textContent = `Patient ${p.id}`;
  document.getElementById('pfacts').textContent =
    `Age ${p.age} · ${p.male ? 'Male' : 'Female'} · last discharge ${p.discharge_date.slice(0, 10)}`;
  const t0 = performance.now();
  const proba = await score(p);
  const ms = Math.round(performance.now() - t0);
  const [cls, label] = band(proba);
  document.getElementById('gscore').textContent = (proba * 100).toFixed(1) + '%';
  const gband = document.getElementById('gband');
  gband.textContent = label + ' risk'; gband.className = 'band ' + cls;
  document.getElementById('gfill').style.width = ((1 - proba) * 100).toFixed(1) + '%';
  document.getElementById('scoredin').textContent =
    `scored in your browser in ${ms} ms · model ${META.model_version}`;

  const maxGain = FINFO.importance[0].gain;
  const box = document.getElementById('factors');
  box.innerHTML = '';
  for (const imp of FINFO.importance.slice(0, 8)) {
    const f = imp.feature, i = META.features.indexOf(f);
    const div = document.createElement('div');
    div.className = 'factor';
    div.innerHTML =
      `<span><strong>${FINFO.display[f]}</strong></span>` +
      `<span class="vals">patient: ${fmt(f, p.features[i])} &nbsp;·&nbsp; median: ${fmt(f, FINFO.medians[f])}</span>` +
      `<span class="bar"><i style="width:${(100 * imp.gain / maxGain).toFixed(1)}%"></i></span>`;
    box.appendChild(div);
  }

  const flagged = proba >= 0.30;
  const hit = (flagged && p.readmitted) || (!flagged && !p.readmitted);
  document.getElementById('outcome').innerHTML =
    `Readmitted within 30 days: <span class="pill ${p.readmitted ? 'yes' : 'no'}">${p.readmitted ? 'Yes' : 'No'}</span>` +
    ` &nbsp;·&nbsp; at a 30% flag threshold the model would have been ` +
    `<span class="pill ${hit ? 'hit' : 'miss'}">${hit ? 'right' : 'wrong'}</span> on this patient.`;
}

function renderModelCard() {
  const dl = document.getElementById('modelcard');
  const rows = [
    ['Champion version', META.model_version],
    ['Features', META.features.length + ' (FHIR-native: demographics, utilization, conditions, vitals/labs)'],
    ['Inference', 'ONNX Runtime Web — runs entirely in your browser'],
    ['Training data', 'Synthea synthetic FHIR R4 (~1,180 patients, fully synthetic)'],
    ['Clinical validity', 'None claimed — systems demo, not a medical device'],
  ];
  dl.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

async function loadStatus() {
  const el = document.getElementById('status');
  let s = null;
  try {
    const r = await fetch('https://raw.githubusercontent.com/webbkyle/readmission-risk/main/data/metrics.json');
    if (r.ok) s = await r.json();
  } catch (e) { /* offline fallback below */ }
  if (!s) { try { s = await fetch('metrics.json').then(r => r.json()); } catch (e) {} }
  if (!s) { el.textContent = 'model status unavailable'; return; }
  const d = s.drift || {};
  el.innerHTML =
    `<strong>Live model status:</strong> champion <strong>${s.champion_version}</strong>` +
    ` · trained on ${s.n_train_rows} discharges / ${s.n_train_patients} patients` +
    ` · holdout AUC ${s.holdout_auc}` +
    ` · last run ${s.last_run} (${s.last_outcome})` +
    (d.max_psi != null ? ` · drift PSI ${d.max_psi} (${d.status})` : '');
}

init().catch(e => {
  document.getElementById('status').textContent = 'failed to load: ' + e.message;
});

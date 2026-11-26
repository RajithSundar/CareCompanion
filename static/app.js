// Every value from the server or the user goes through esc() before it is put into innerHTML.
function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

// fetch wrapper: JSON in and out, throws an Error with the server's message on failure.
async function api(path, { method = 'GET', body } = {}) {
  const opts = { method, headers: {}, credentials: 'same-origin' };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    throw new Error('Could not reach the server. Check your connection.');
  }
  const data = await res.json().catch(() => ({}));
  if (res.status === 401 && !path.startsWith('/api/auth/')) {
    // Session expired or signed out in another tab: back to the sign-in page, then here.
    location.assign(`/login?next=${encodeURIComponent(location.pathname)}`);
    throw new Error('Signed out. Sign in again to continue.');
  }
  if (!res.ok) {
    let msg = data.detail;
    if (Array.isArray(msg)) msg = msg.map(d => `${(d.loc || []).slice(-1)[0] || 'field'}: ${d.msg}`).join('; ');
    if (res.status === 429) msg = msg || 'Too many requests. Wait a moment and try again.';
    throw new Error(msg || `Server error (HTTP ${res.status})`);
  }
  return data;
}

const pid = () => state.activePersonaId.toLowerCase();

// Each page has only some of the dashboard's elements. $() returns a detached stand-in
// for ones that are missing, so shared render code can run on every page unchanged.
function $(id) {
  return document.getElementById(id) || document.createElement('div');
}
const has = id => Boolean(document.getElementById(id));
const PAGE = document.body.dataset.page || 'home';

// The signed-in account, rendered into the header by the server. Null when signed out.
const ACCOUNT = (() => {
  const el = document.getElementById('accountChip');
  return el ? { pid: el.dataset.pid, name: el.dataset.name } : null;
})();

// The chosen session follows the visitor from page to page.
const store = {
  get(k, fallback) { try { const v = localStorage.getItem(`cc_${k}`); return v === null ? fallback : v; } catch (e) { return fallback; } },
  set(k, v) { try { localStorage.setItem(`cc_${k}`, String(v)); } catch (e) {} }
};

function getToken(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Filled from /api/model-info (permutation importance). Empty until the server responds.
let FEATURE_IMPORTANCES = [];

const FALLBACK_PERSONA = {
  id: 'P01', name: 'Alex Rivera', role: 'Half-Marathon Prep, Morning Runner',
  baseline: { steps: 9400, rhr: 58, sleepScore: 79, readiness: 7 },
  days: [
    { day: 'Day 1', steps: 9650, rhr: 58, sleepScore: 80, deepMin: 88, remMin: 95, readiness: 8, fatigue: 2, soreness: 2, stress: 2, fatBurn: 38, cardio: 26, peak: 8 }
  ],
  sessions: [
    {
      id: 'S1', label: 'Session 1', dayIndex: 0, dayLabel: 'Day 1',
      title: 'Session 1: Baseline Intake', subtitle: 'Establishing biometric baselines and half-marathon goal',
      adaptation: 'Captured half-marathon goal, morning preference, and left knee Peak HR cap.',
      plan: [{ id: 'p1_1', text: '35 min Zone 2/3 morning run', done: true }],
      messages: [{ role: 'coach', text: 'Baseline metrics loaded.', citations: ['Baseline Recorded'] }]
    }
  ],
  memories: [
    { id: 'm1', category: 'Goal', text: 'Training for half-marathon in 8 weeks', session: 'S1', activeFrom: 0 }
  ],
  evolutionLog: [
    { session: 'S1', text: 'Captured half-marathon goal and baseline.' }
  ]
};

const state = {
  personas: { P01: JSON.parse(JSON.stringify(FALLBACK_PERSONA)) },
  activePersonaId: ACCOUNT ? ACCOUNT.pid : 'P01', // the account's own profile
  activeSessionIndex: null, // restored per profile once its sessions load
  chartMode: 'activity',
  serverLlm: null, // { provider, model } when the server has a key
  loadSeq: 0,
  chartInstance: null,
  isSending: false,
  modelInfo: null,
  livePrediction: undefined // undefined = loading, null = unavailable
};

// Older versions stored an API key and a chosen profile in the browser. Remove them.
try { localStorage.removeItem('carecompanion_api_v2'); localStorage.removeItem('cc_persona'); } catch (e) {}

function getPersona() { return state.personas[state.activePersonaId] || FALLBACK_PERSONA; }
function getSession() {
  const p = getPersona();
  return p.sessions[Math.min(state.activeSessionIndex, p.sessions.length - 1)] || p.sessions[0];
}
function getActiveDayRecord() {
  const p = getPersona();
  const s = getSession();
  return p.days[Math.min(s.dayIndex, p.days.length - 1)] || p.days[0];
}

async function loadParticipantState(pid) {
  const seq = ++state.loadSeq;
  try {
    const data = await api(`/api/state/${encodeURIComponent(pid.toLowerCase())}`);
    if (seq !== state.loadSeq) return; // a newer persona switch won
    state.profileLoaded = true;
    state.personas[pid] = {
      id: data.id,
      name: data.name,
      role: data.role,
      baseline: data.baseline,
      days: data.days,
      sessions: data.sessions,
      evolutionLog: data.evolutionLog || [],
      memories: (data.memories || []).map(m => ({ ...m, activeFrom: 0 }))
    };
    const count = state.personas[pid].sessions.length;
    const saved = Number(store.get(`session_${pid}`, NaN));
    if (state.activeSessionIndex === null || state.activeSessionIndex >= count) {
      state.activeSessionIndex = Number.isInteger(saved) && saved >= 0 && saved < count ? saved : count - 1;
    }
    renderAll();
    syncPredictionFromBackend();
    sendPendingPrompt();
  } catch (e) {
    if (seq === state.loadSeq) {
      showToast(`Could not load profile: ${e.message}`);
      if (!state.profileLoaded && window.CCTracker) window.CCTracker.render(null);
    }
  }
}

async function loadModelInfo() {
  try {
    const data = await api('/api/model-info');
    state.serverLlm = data.llm_configured ? { provider: data.llm_provider, model: data.llm_model } : null;
    state.modelInfoLoaded = true;
    renderApiStatus();
    if (data.trained && data.feature_importances) {
      state.modelInfo = data;
      FEATURE_IMPORTANCES = Object.entries(data.feature_importances)
        .slice(0, 6)
        .map(([name, v]) => ({ name, val: Math.max(0, v.mean) }));
      renderOverview();
    }
  } catch (e) {}
}

async function syncPredictionFromBackend() {
  const p = getPersona();
  const s = getSession();
  const key = `${p.id}:${s.id}:${s.dayIndex}`;
  state.predictionKey = key;
  try {
    const data = await api(`/api/predict/${encodeURIComponent(p.id.toLowerCase())}?day_index=${s.dayIndex}`);
    if (state.predictionKey !== key) return;
    state.livePrediction = {
      label: data.predicted_recovery_state,
      probs: data.probabilities,
      modelName: data.model_name
    };
  } catch (e) {
    if (state.predictionKey !== key) return;
    state.livePrediction = null;
  }
  renderOverview();
  renderUpdateInputs();
}

function showToast(msg) {
  const toast = $('toast');
  $('toastText').textContent = msg;
  toast.classList.remove('hidden'); toast.classList.add('flex');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.classList.add('hidden'); toast.classList.remove('flex'); }, 3200);
}

function formatMarkdownSimple(text) {
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong class="font-medium text-accent">$1</strong>').replace(/\n/g, '<br>');
}

function renderAll() {
  renderOverview();
  renderLedger();
  renderSessionStepper();
  renderKPIs();
  renderUpdateInputs();
  renderChart();
  renderChatHeaderAndPlan();
  renderMessages();
  renderQuickPrompts();
  renderMemoryPanel();
  renderApiStatus();
  if (window.lucide) lucide.createIcons();
}

// Feeds the hero tracker (tracker.js). Until a profile loads it shows a labelled random sample.
function renderTracker() {
  if (!window.CCTracker) return;
  if (!state.profileLoaded) return; // tracker.js falls back to a random sample if no profile arrives
  const p = getPersona(); const d = getActiveDayRecord(); const pred = state.livePrediction;
  window.CCTracker.render({
    sample: false,
    who: p.name.split(' ')[0],
    day: d.day,
    steps: d.steps,
    rhr: d.rhr,
    sleep: d.sleepScore,
    readiness: d.readiness,
    forecast: pred ? pred.label : null
  });
}

function renderOverview() {
  renderTracker();
  const p = getPersona();
  const s = getSession();
  const d = getActiveDayRecord();
  const pred = state.livePrediction;

  $('overviewSessionBadge').textContent = `${s.label}, ${s.dayLabel}`;
  $('overviewRole').textContent = p.role;
  $('overviewGreeting').textContent = p.name;
  $('overviewAdaptation').textContent = s.adaptation;

  const switcherEl = $('overviewSessionSwitcher');
  switcherEl.innerHTML = p.sessions.map((sess, idx) => {
    const active = idx === state.activeSessionIndex;
    return `<button type="button" data-overview-sess="${idx}" class="overview-sess-btn flex-1 min-w-[96px] py-2 px-2 text-xs font-medium text-center whitespace-nowrap btn-transition ${
      active
        ? 'bg-cool-soft text-cool'
        : 'bg-paper-3 text-muted hover:text-ink'
    }">${esc(sess.id)}: ${esc(sess.dayLabel)}</button>`;
  }).join('');
  switcherEl.querySelectorAll('.overview-sess-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      selectSession(Number(btn.dataset.overviewSess));
    });
  });

  const clsEl = $('mlPredClass');
  clsEl.textContent = pred === undefined ? 'Loading…' : pred === null ? 'Unavailable' : pred.label;
  clsEl.className = `forecast-label ${
    !pred ? 'text-muted' : pred.label === 'Ready' ? 'text-cool' : pred.label === 'Deload Needed' ? 'text-accent' : 'text-ink'
  }`;

  $('mlProbPills').innerHTML = !pred ? '' : ['Ready', 'Moderate', 'Deload Needed'].filter(k => k in pred.probs).map(k => {
    const pct = Math.round(pred.probs[k] * 100);
    return `<div class="prob-row${k === pred.label ? ' is-top' : ''}">
      <span class="prob-name">${esc(k)}</span>
      <span class="prob-track"><span class="prob-fill" style="transform: scaleX(${(pct / 100).toFixed(3)})"></span></span>
      <span class="prob-pct">${pct}%</span>
    </div>`;
  }).join('');

  if (state.modelInfo && state.modelInfo.selected_model) {
    $('modelNameBadge').textContent = state.modelInfo.selected_model;
  }

  const maxImp = Math.max(...FEATURE_IMPORTANCES.map(x => x.val), 0.01);
  $('featureImportanceBars').innerHTML = FEATURE_IMPORTANCES.map((f, idx) => {
    const pct = Math.min(100, Math.round((f.val / maxImp) * 100));
    const barColor = idx === 0 ? 'bg-accent' : 'bg-cool';
    return `
      <div class="flex items-center gap-3 text-xs">
        <span class="w-36 truncate text-ink-2">${esc(f.name)}</span>
        <div class="flex-1 h-2 bg-paper-3 border border-rule rounded-full overflow-hidden">
          <div class="h-full ${barColor} rounded-full" style="width: ${pct}%"></div>
        </div>
        <span class="w-16 text-right measured text-ink">${f.val.toFixed(3)}</span>
      </div>
    `;
  }).join('');

  const compRows = state.modelInfo && Array.isArray(state.modelInfo.comparison) ? state.modelInfo.comparison : [];
  const baselines = state.modelInfo && state.modelInfo.baselines ? state.modelInfo.baselines : {};
  const selected = (state.modelInfo && state.modelInfo.selected_model) || '';
  const rows = [
    ...compRows.map(row => ({ name: row.model, f1: row.macro_f1, acc: row.accuracy, best: row.model === selected, baseline: false })),
    ...Object.entries(baselines).map(([k, b]) => ({ name: k.replace(/_/g, ' '), f1: b.macro_f1, acc: b.accuracy, best: false, baseline: true }))
  ];
  $('modelComparisonGrid').innerHTML = rows.length
    ? rows.map(row => `
      <div class="p-2.5 rounded border ${row.best ? 'bg-accent-soft border-accent' : 'bg-paper-2 border-rule'}">
        <div class="text-xs font-medium ${row.best ? 'text-accent' : 'text-muted'}">${esc(row.name)}${row.baseline ? ' (baseline)' : ''}</div>
        <div class="text-sm measured mt-0.5 ${row.best ? 'text-ink' : 'text-ink-2'}">Macro F1 ${(row.f1 * 100).toFixed(1)}%, accuracy ${(row.acc * 100).toFixed(1)}%</div>
      </div>
    `).join('')
    : '<p class="text-xs text-muted">Model metrics unavailable. Run train_model.py.</p>';
}

function renderLedger() {
  const p = getPersona();
  const s = getSession();
  const body = $('biometricLedgerBody');
  body.innerHTML = p.days.map((d, idx) => {
    const isSelected = idx === s.dayIndex;
    return `
      <tr class="${isSelected ? 'bg-cool-soft text-ink' : 'hover:bg-paper-2'}">
        <td class="py-2 px-3 font-sans font-medium whitespace-nowrap ${isSelected ? 'text-cool' : 'text-ink'}">${esc(d.day)}</td>
        <td class="py-2 px-2.5 whitespace-nowrap">${Number(d.steps).toLocaleString()}</td>
        <td class="py-2 px-2.5 whitespace-nowrap">${esc(d.rhr)} bpm</td>
        <td class="py-2 px-2.5 whitespace-nowrap">${esc(d.sleepScore)}</td>
        <td class="py-2 px-2.5 whitespace-nowrap">${esc(d.readiness)}/10</td>
        <td class="py-2 px-2.5 whitespace-nowrap">${esc(d.fatigue)}/5</td>
        <td class="py-2 px-2.5 whitespace-nowrap">${esc(d.soreness)}/5</td>
        <td class="py-2 px-3 whitespace-nowrap">${esc(d.stress)}/5</td>
      </tr>
    `;
  }).join('');
}

function renderSessionStepper() {
  const p = getPersona();
  const container = $('sessionStepper');
  container.innerHTML = p.sessions.map((s, idx) => {
    const active = idx === state.activeSessionIndex;
    return `
      <button type="button" data-session-idx="${idx}" class="session-step-btn w-full flex items-center justify-between px-4 py-3 text-left btn-transition ${
        active ? 'bg-accent-soft text-ink' : 'bg-paper-3 text-muted hover:text-ink hover:bg-paper-2'
      }">
        <div class="flex-1 min-w-0 pr-2">
          <div class="text-xs font-medium ${active ? 'text-accent' : 'text-ink'}">${esc(s.label)}: ${esc(s.dayLabel)}</div>
          <div class="text-xs truncate text-muted mt-0.5">${esc(s.subtitle)}</div>
        </div>
        <span class="measured text-xs px-2 py-0.5 rounded shrink-0 ${
          active ? 'bg-accent text-accent-ink font-medium' : 'bg-paper-2 text-muted border border-rule'
        }">${idx + 1}</span>
      </button>
    `;
  }).join('');
  container.querySelectorAll('.session-step-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      selectSession(Number(btn.dataset.sessionIdx));
    });
  });
}

function renderKPIs() {
  const p = getPersona(); const d = getActiveDayRecord(); const base = p.baseline;
  $('kpiSteps').textContent = Number(d.steps).toLocaleString();
  const stepDiff = d.steps - base.steps;
  const stepEl = $('kpiStepsDelta');
  stepEl.textContent = `${stepDiff >= 0 ? '+' : ''}${stepDiff.toLocaleString()}`;
  stepEl.className = `col-span-3 text-right measured text-xs ${stepDiff >= 0 ? 'text-cool' : 'text-accent'}`;

  $('kpiRhr').textContent = `${d.rhr} bpm`;
  const rhrDiff = d.rhr - base.rhr;
  const rhrEl = $('kpiRhrDelta');
  rhrEl.textContent = `${rhrDiff >= 0 ? '+' : ''}${rhrDiff} bpm`;
  rhrEl.className = `col-span-3 text-right measured text-xs ${rhrDiff >= 4 ? 'text-accent' : rhrDiff <= 0 ? 'text-cool' : 'text-muted'}`;

  $('kpiSleep').textContent = `${d.sleepScore}/100`;
  const sleepDiff = d.sleepScore - base.sleepScore;
  const sleepEl = $('kpiSleepDelta');
  sleepEl.textContent = `${sleepDiff >= 0 ? '+' : ''}${sleepDiff}`;
  sleepEl.className = `col-span-3 text-right measured text-xs ${sleepDiff >= 0 ? 'text-cool' : 'text-accent'}`;

  $('kpiReadiness').textContent = `${d.readiness}/10`;
  const readEl = $('kpiReadinessDelta');
  readEl.textContent = d.readiness >= 7 ? 'Ready' : d.readiness >= 5 ? 'Moderate' : 'Deload';
  readEl.className = `col-span-3 text-right text-xs font-medium ${d.readiness >= 7 ? 'text-cool' : 'text-accent'}`;
}

function renderUpdateInputs() {
  const pred = state.livePrediction;
  const d = getActiveDayRecord();
  $('inputSteps').value = d.steps;
  $('inputRhr').value = d.rhr;
  $('inputSleep').value = d.sleepScore;
  $('inputDeep').value = d.deepMin;
  $('inputRem').value = d.remMin;
  $('inputCardio').value = d.cardio;
  $('inputPeak').value = d.peak;
  $('checkinTargetDay').textContent = d.day;
  $('sliderReadiness').value = d.readiness;
  $('valReadiness').textContent = d.readiness;
  $('sliderFatigue').value = d.fatigue;
  $('valFatigue').textContent = d.fatigue;
  $('sliderSoreness').value = d.soreness;
  $('valSoreness').textContent = d.soreness;
  $('sliderStress').value = d.stress;
  $('valStress').textContent = d.stress;

  const badge = $('liveUpdateBadge');
  badge.textContent = pred ? `Forecast: ${pred.label}` : 'Forecast unavailable';
  badge.className = `px-2.5 py-1 rounded text-xs font-medium border ${
    pred && pred.label === 'Deload Needed'
      ? 'bg-accent-soft text-accent border-accent'
      : 'bg-cool-soft text-cool border-rule-2'
  }`;
}

function renderChart() {
  if (!has('biometricChart') || !window.Chart) return;
  const p = getPersona();
  const visibleDays = p.days;
  const labels = visibleDays.map(d => d.day);
  const ctx = $('biometricChart').getContext('2d');
  if (state.chartInstance) state.chartInstance.destroy();

  const cool = getToken('--color-cool');
  const accent = getToken('--color-accent');
  const ink = getToken('--color-ink');
  const muted = getToken('--color-muted');
  const rule = getToken('--color-rule');
  const paper = getToken('--color-paper');

  let datasets = [], scales = {};
  if (state.chartMode === 'activity') {
    datasets = [
      {
        label: 'Steps',
        data: visibleDays.map(d => d.steps),
        backgroundColor: cool,
        borderColor: cool,
        borderWidth: 1,
        type: 'bar',
        borderRadius: 4,
        yAxisID: 'y'
      },
      {
        label: 'Sleep score',
        data: visibleDays.map(d => d.sleepScore),
        borderColor: accent,
        backgroundColor: accent,
        pointBackgroundColor: accent,
        pointBorderColor: paper,
        pointBorderWidth: 1.5,
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 3.5,
        type: 'line',
        yAxisID: 'y1'
      }
    ];
    scales = {
      y: { position: 'left', beginAtZero: true, grid: { color: rule }, ticks: { color: muted } },
      y1: { position: 'right', min: 40, max: 100, grid: { drawOnChartArea: false }, ticks: { color: accent } },
      x: { grid: { display: false }, ticks: { color: muted } }
    };
  } else if (state.chartMode === 'recovery') {
    datasets = [
      {
        label: 'Resting HR (bpm)',
        data: visibleDays.map(d => d.rhr),
        borderColor: accent,
        backgroundColor: accent,
        pointBackgroundColor: accent,
        borderWidth: 2,
        tension: 0.25,
        yAxisID: 'y'
      },
      {
        label: 'PMSys readiness',
        data: visibleDays.map(d => d.readiness),
        borderColor: cool,
        backgroundColor: cool,
        pointBackgroundColor: cool,
        borderWidth: 2,
        tension: 0.25,
        yAxisID: 'y1'
      }
    ];
    scales = {
      y: { position: 'left', min: 45, max: 80, grid: { color: rule }, ticks: { color: accent } },
      y1: { position: 'right', min: 0, max: 10, grid: { drawOnChartArea: false }, ticks: { color: cool } },
      x: { grid: { display: false }, ticks: { color: muted } }
    };
  } else {
    datasets = [
      { label: 'Fat burn (min)', data: visibleDays.map(d => (d.fatBurn ?? null)), backgroundColor: cool, stack: 'hr' },
      { label: 'Cardio (min)', data: visibleDays.map(d => d.cardio), backgroundColor: accent, stack: 'hr' },
      { label: 'Peak (min)', data: visibleDays.map(d => d.peak), backgroundColor: ink, stack: 'hr' }
    ];
    scales = {
      y: { stacked: true, beginAtZero: true, grid: { color: rule }, ticks: { color: muted } },
      x: { stacked: true, grid: { display: false }, ticks: { color: muted } }
    };
  }

  state.chartInstance = new Chart(ctx, {
    type: state.chartMode === 'zones' ? 'bar' : 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { boxWidth: 10, color: ink, font: { size: 11 } }
        }
      },
      scales
    }
  });

  document.querySelectorAll('.chart-mode-btn').forEach(btn => {
    const active = btn.dataset.chartMode === state.chartMode;
    btn.className = `chart-mode-btn px-3 py-1.5 rounded text-xs font-medium whitespace-nowrap btn-transition ${
      active ? 'bg-accent text-accent-ink' : 'text-muted hover:text-ink'
    }`;
  });
}

function renderChatHeaderAndPlan() {
  const p = getPersona(); const s = getSession();
  $('sessionTitle').textContent = s.title;
  $('sessionDayBadge').textContent = s.dayLabel;
  $('sessionSubtitle').textContent = s.subtitle;

  const nextBtn = $('nextSessionBtn');
  if (state.activeSessionIndex < p.sessions.length - 1) {
    nextBtn.classList.remove('hidden'); nextBtn.classList.add('inline-flex');
  } else {
    nextBtn.classList.add('hidden'); nextBtn.classList.remove('inline-flex');
  }

  const doneCount = s.plan.filter(item => item.done).length;
  $('planAdherenceBadge').textContent = `${s.plan.length ? Math.round((doneCount / s.plan.length) * 100) : 0}%`;

  const planList = $('actionPlanList');
  planList.innerHTML = s.plan.map(item => `
    <label class="flex items-center gap-2.5 p-2.5 rounded border text-xs cursor-pointer btn-transition ${
      item.done
        ? 'bg-cool-soft border-rule-2 text-ink'
        : 'bg-paper-2 border-rule text-ink-2 hover:border-cool'
    }">
      <input type="checkbox" data-plan-id="${esc(item.id)}" ${item.done ? 'checked' : ''} class="plan-checkbox rounded range-cool cursor-pointer" />
      <span class="${item.done ? 'line-through text-muted' : ''}">${esc(item.text)}</span>
    </label>
  `).join('');

  planList.querySelectorAll('.plan-checkbox').forEach(cb => {
    cb.addEventListener('change', async () => {
      const target = s.plan.find(x => x.id === cb.dataset.planId);
      if (!target) return;
      const previous = target.done;
      target.done = cb.checked;
      renderChatHeaderAndPlan();
      try {
        await api(`/api/sessions/${encodeURIComponent(p.id.toLowerCase())}/${encodeURIComponent(s.id)}/plan/${encodeURIComponent(target.id)}`,
          { method: 'PATCH', body: { done: target.done } });
      } catch (e) {
        target.done = previous;
        renderChatHeaderAndPlan();
        showToast(`Could not save: ${e.message}`);
      }
    });
  });
}

function renderMessages() {
  const s = getSession();
  const container = $('chatMessages');
  const all = state.pendingSessionId === s.id ? [...s.messages, { role: 'pending' }] : s.messages;
  container.innerHTML = all.map(m => {
    if (m.role === 'pending') {
      return `<div class="flex items-start gap-3" aria-live="polite"><div class="w-8 h-8 rounded-md bg-paper-2 border border-rule-2 flex items-center justify-center shrink-0 mt-0.5" aria-hidden="true"><i data-lucide="loader" class="w-4 h-4 text-muted animate-spin"></i></div><div class="rounded-md bg-paper-2 border border-rule px-4 py-3 text-sm text-muted">The coach is thinking…</div></div>`;
    }
    if (m.role === 'error') {
      return `<div class="flex items-start gap-3" role="alert"><div class="w-8 h-8 rounded-md bg-accent-soft border border-accent flex items-center justify-center shrink-0 mt-0.5" aria-hidden="true"><i data-lucide="alert-triangle" class="w-4 h-4 text-accent"></i></div><div class="rounded-md bg-accent-soft border border-accent px-4 py-3 text-sm text-ink">${esc(m.text)} This reply was not saved.</div></div>`;
    }
    if (m.role === 'user') {
      return `<div class="flex justify-end"><div class="max-w-[82%] rounded-md bg-paper-3 border border-accent text-ink px-4 py-2.5 text-sm">${formatMarkdownSimple(m.text)}</div></div>`;
    }
    const citationsHtml = (m.citations && m.citations.length)
      ? `<div class="mt-2.5 flex flex-wrap gap-1.5">${m.citations.map(c => `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs bg-cool-soft text-cool border border-rule-2">${esc(c)}</span>`).join('')}</div>`
      : '';
    return `
      <div class="flex items-start gap-3">
        <div class="w-8 h-8 rounded-md bg-cool text-accent-ink flex items-center justify-center shrink-0 mt-0.5" aria-hidden="true"><i data-lucide="activity" class="w-4 h-4"></i></div>
        <div class="flex-1 min-w-0">
          <div class="rounded-md bg-paper-3 border border-rule-2 px-4 py-3 text-sm text-ink-2 leading-relaxed">${formatMarkdownSimple(m.text)}${citationsHtml}</div>
        </div>
      </div>
    `;
  }).join('');
  container.scrollTop = container.scrollHeight;
  if (window.lucide) lucide.createIcons();
}

function renderQuickPrompts() {
  const prompts = ['What should my workout intensity be today?', 'How is my sleep affecting recovery?', 'Summarize what you remember about me'];
  const container = $('quickPrompts');
  container.innerHTML = prompts.map(text => `<button type="button" class="quick-prompt-btn px-3 py-1 rounded bg-paper-2 border border-rule-2 hover:border-cool text-xs text-ink-2 hover:text-ink whitespace-nowrap btn-transition">${text}</button>`).join('');
  container.querySelectorAll('.quick-prompt-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $('chatInput').value = btn.textContent.trim();
      $('chatForm').dispatchEvent(new Event('submit'));
    });
  });
}

function renderMemoryPanel() {
  const p = getPersona();
  const catColors = {
    Goal: 'bg-cool-soft text-cool border-rule-2',
    Constraint: 'bg-accent-soft text-accent border-accent',
    Baseline: 'bg-paper-2 text-ink-2 border-rule-2',
    Adaptation: 'bg-accent text-accent-ink border-accent'
  };
  const listEl = $('memoryList');
  listEl.innerHTML = p.memories.map(m => `
    <div class="p-3 rounded-md border border-rule bg-paper-3 hover:border-rule-2 btn-transition group">
      <div class="flex items-center justify-between gap-2 mb-1.5">
        <span class="px-2 py-0.5 rounded text-xs font-medium border ${catColors[m.category] || catColors.Baseline}">${esc(m.category === 'Adaptation' ? 'Rule' : m.category)}</span>
        <div class="flex items-center gap-2">
          <span class="text-xs text-muted">${esc(m.session)}</span>
          <button type="button" data-del-mem="${esc(m.id)}" aria-label="Delete memory" class="del-mem-btn opacity-0 group-hover:opacity-100 focus:opacity-100 text-muted hover:text-accent"><i data-lucide="trash-2" class="w-3.5 h-3.5" aria-hidden="true"></i></button>
        </div>
      </div>
      <p class="text-xs text-ink-2 leading-relaxed break-words">${esc(m.text)}</p>
    </div>
  `).join('');
  listEl.querySelectorAll('.del-mem-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const memId = btn.dataset.delMem;
      btn.disabled = true;
      try {
        const data = await api(`/api/memory/${encodeURIComponent(p.id.toLowerCase())}/${encodeURIComponent(memId)}`, { method: 'DELETE' });
        p.memories = data.memories.map(m => ({ ...m, activeFrom: 0 }));
        renderMemoryPanel();
        if (window.lucide) lucide.createIcons();
        showToast('Memory removed');
      } catch (e) {
        btn.disabled = false;
        showToast(`Could not remove memory: ${e.message}`);
      }
    });
  });

  $('memoryEvolutionLog').innerHTML = p.evolutionLog.map(l => `
    <div class="flex items-start gap-3 p-3">
      <span class="px-2 py-0.5 rounded bg-cool-soft text-cool font-medium shrink-0">${esc(l.session)}</span>
      <span class="text-ink-2 leading-relaxed">${esc(l.text)}</span>
    </div>
  `).join('');
}

function renderApiStatus() {
  const label = $('apiStatusLabel');
  const detail = $('coachStatusText');
  if (!state.modelInfoLoaded) {
    label.textContent = 'Coach status';
    detail.textContent = 'Checking the server…';
  } else if (state.serverLlm) {
    label.textContent = `Coach: ${state.serverLlm.provider}`;
    detail.textContent = `The coach writes replies with ${state.serverLlm.provider} (${state.serverLlm.model}), grounded in the forecast and your saved memories.`;
  } else {
    label.textContent = 'Coach: offline replies';
    detail.textContent = 'No language model is configured on the server, so the coach gives rule-based replies built from the forecast and your saved memories.';
  }
}

// Sends a message to the coach. The server saves both turns to the session on success.
// Returns true on success. On failure the user turn and an error note are shown but not saved.
async function sendToCoach(userText) {
  const p = getPersona(); const s = getSession();
  if (state.pendingSessionId) return false;
  const history = s.messages.filter(m => m.role === 'user' || m.role === 'coach').slice(-6).map(m => ({ role: m.role, text: String(m.text).slice(0, 4000) }));
  s.messages.push({ role: 'user', text: userText });
  state.pendingSessionId = s.id;
  setChatBusy(true);
  renderMessages();
  try {
    const data = await api('/api/coach', {
      method: 'POST',
      body: {
        participant_id: p.id.toLowerCase(),
        session_id: s.id,
        message: userText,
        day_index: s.dayIndex,
        history,
        auto_extract_memory: true
      }
    });
    if (data.ml_prediction) {
      state.livePrediction = {
        label: data.ml_prediction.predicted_recovery_state,
        probs: data.ml_prediction.probabilities,
        modelName: data.ml_prediction.model_name
      };
    }
    if (Array.isArray(data.memories)) p.memories = data.memories.map(m => ({ ...m, activeFrom: 0 }));
    s.messages.push({ role: 'coach', text: data.reply, citations: data.citations || [] });
    if (Array.isArray(data.newly_extracted_memories) && data.newly_extracted_memories.length > 0) {
      showToast(`Saved ${data.newly_extracted_memories.length} new memory fact(s)`);
    }
    return true;
  } catch (err) {
    s.messages.push({ role: 'error', text: err.message });
    return false;
  } finally {
    state.pendingSessionId = null;
    setChatBusy(false);
    renderMessages();
    renderMemoryPanel();
    renderOverview();
    if (window.lucide) lucide.createIcons();
  }
}

function sendPendingPrompt() {
  if (PAGE !== 'coach') return;
  let pending = null;
  try { pending = JSON.parse(sessionStorage.getItem('cc_pending_prompt') || 'null'); sessionStorage.removeItem('cc_pending_prompt'); } catch (e) {}
  if (!pending || pending.persona !== state.activePersonaId) return;
  const idx = getPersona().sessions.findIndex(x => x.id === pending.session);
  if (idx >= 0 && idx !== state.activeSessionIndex) selectSession(idx);
  sendToCoach(pending.text);
}

function setChatBusy(busy) {
  const input = $('chatInput');
  const btn = $('sendChatBtn');
  input.disabled = busy;
  btn.disabled = busy;
  btn.classList.toggle('opacity-60', busy);
  document.querySelectorAll('.quick-prompt-btn').forEach(b => { b.disabled = busy; });
  if (!busy) input.focus({ preventScroll: true });
}

function selectSession(idx) {
  state.activeSessionIndex = idx;
  store.set(`session_${state.activePersonaId}`, idx);
  state.livePrediction = undefined;
  renderAll();
  syncPredictionFromBackend();
}

$('logoutBtn').addEventListener('click', async (e) => {
  e.currentTarget.disabled = true;
  try { await api('/api/auth/logout', { method: 'POST', body: {} }); } catch (err) {}
  try { sessionStorage.removeItem('cc_pending_prompt'); } catch (err) {}
  location.assign('/');
});

$('nextSessionBtn').addEventListener('click', () => {
  const p = getPersona();
  if (state.activeSessionIndex < p.sessions.length - 1) selectSession(state.activeSessionIndex + 1);
});

$('newSessionBtn').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  const p = getPersona();
  btn.disabled = true;
  try {
    const data = await api(`/api/sessions/${encodeURIComponent(p.id.toLowerCase())}`, { method: 'POST', body: {} });
    p.sessions = data.sessions;
    selectSession(p.sessions.length - 1);
    showToast(`Opened ${data.session.label}`);
  } catch (err) {
    showToast(`Could not open a session: ${err.message}`);
  } finally {
    btn.disabled = false;
  }
});

document.querySelectorAll('.chart-mode-btn').forEach(btn => {
  btn.addEventListener('click', () => { state.chartMode = btn.dataset.chartMode; renderChart(); });
});

['Readiness', 'Fatigue', 'Soreness', 'Stress'].forEach(name => {
  const slider = $(`slider${name}`);
  const valEl = $(`val${name}`);
  slider.addEventListener('input', () => { valEl.textContent = slider.value; });
});

function showCheckinError(msg) {
  const el = $('checkinError');
  el.textContent = msg || '';
  el.classList.toggle('hidden', !msg);
}

$('applyCheckinBtn').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  const p = getPersona();
  const s = getSession();
  const mode = document.querySelector('input[name="checkinMode"]:checked').value;
  const num = id => {
    const raw = $(id).value;
    return raw === '' ? NaN : Number(raw);
  };
  const payload = {
    mode,
    session_id: s.id,
    steps: num('inputSteps'),
    rhr: num('inputRhr'),
    sleepScore: num('inputSleep'),
    deepMin: num('inputDeep'),
    remMin: num('inputRem'),
    cardio: num('inputCardio'),
    peak: num('inputPeak'),
    readiness: num('sliderReadiness'),
    fatigue: num('sliderFatigue'),
    soreness: num('sliderSoreness'),
    stress: num('sliderStress')
  };
  if (mode === 'update') payload.dayIndex = s.dayIndex;
  const blank = Object.entries(payload).filter(([, v]) => typeof v === 'number' && Number.isNaN(v)).map(([k]) => k);
  if (blank.length) { showCheckinError(`Fill in: ${blank.join(', ')}.`); return; }

  showCheckinError('');
  btn.disabled = true;
  let data;
  try {
    data = await api(`/api/checkin/${encodeURIComponent(p.id.toLowerCase())}`, { method: 'POST', body: payload });
  } catch (err) {
    showCheckinError(`Check-in not saved: ${err.message}`);
    btn.disabled = false;
    return;
  }
  p.days = data.days;
  if (data.session) Object.assign(s, data.session);
  state.livePrediction = data.ml_prediction ? {
    label: data.ml_prediction.predicted_recovery_state,
    probs: data.ml_prediction.probabilities,
    modelName: data.ml_prediction.model_name
  } : null;
  renderAll();
  const pred = state.livePrediction;
  showToast(pred ? `Saved ${data.day_record.day} (forecast for the next day: ${pred.label})` : `Saved ${data.day_record.day}`);

  const d = data.day_record;
  const prompt = `I just logged ${d.day}: ${d.steps} steps, RHR ${d.rhr} bpm, sleep ${d.sleepScore}/100, readiness ${d.readiness}/10, fatigue ${d.fatigue}/5, soreness ${d.soreness}/5, stress ${d.stress}/5. How should I adjust my plan?`;
  btn.disabled = false;
  $('checkinResultTitle').textContent = `Saved ${d.day}`;
  $('checkinResultText').textContent = pred ? `Forecast for the next day: ${pred.label}.` : 'The forecast is unavailable right now.';
  $('checkinResult').classList.remove('hidden');
  $('checkinAskCoach').onclick = () => {
    try { sessionStorage.setItem('cc_pending_prompt', JSON.stringify({ persona: p.id, session: s.id, text: prompt })); } catch (e) {}
  };
});

$('chatForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = $('chatInput');
  const text = input.value.trim();
  if (!text || state.pendingSessionId) return;
  if (text.length > 2000) { showToast('Keep messages under 2,000 characters.'); return; }
  input.value = '';
  await sendToCoach(text);
});

$('addMemoryForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const textInput = $('newMemoryText');
  const text = textInput.value.trim();
  if (!text) return;
  if (text.length > 280) { showToast('Keep memories under 280 characters.'); return; }
  const category = $('newMemoryCategory').value;
  const session = (getSession().id || 'Live').slice(0, 20);
  try {
    const data = await api(`/api/memory/${encodeURIComponent(pid())}`, { method: 'POST', body: { category, text, session } });
    getPersona().memories = data.memories.map(m => ({ ...m, activeFrom: 0 }));
    textInput.value = '';
    renderMemoryPanel();
    if (window.lucide) lucide.createIcons();
    showToast('Memory saved');
  } catch (err) {
    showToast(`Memory not saved: ${err.message}`);
  }
});

$('resetDemoBtn').addEventListener('click', async () => {
  if (!confirm('Reset this profile to the demo data? Your check-ins, sessions and memories for it will be lost.')) return;
  try {
    await api(`/api/state/${encodeURIComponent(pid())}/reset`, { method: 'POST', body: {} });
    state.activeSessionIndex = 0;
    state.livePrediction = undefined;
    await loadParticipantState(state.activePersonaId);
    showToast('Demo data restored');
  } catch (err) {
    showToast(`Reset failed: ${err.message}`);
  }
});

const apiModal = $('apiModal');
function closeApiModal() { apiModal.classList.add('hidden'); apiModal.classList.remove('flex'); }
$('openApiModalBtn').addEventListener('click', () => {
  renderApiStatus();
  apiModal.classList.remove('hidden'); apiModal.classList.add('flex');
  $('closeApiModalBtn').focus();
});
$('closeApiModalBtn').addEventListener('click', closeApiModal);
apiModal.addEventListener('click', (e) => { if (e.target === apiModal) closeApiModal(); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !apiModal.classList.contains('hidden')) closeApiModal();
});

const IMPORT_REQUIRED = ['steps', 'rhr', 'sleepScore', 'deepMin', 'remMin', 'cardio', 'peak', 'readiness', 'fatigue', 'soreness', 'stress'];

function parseImport(raw) {
  if (raw.startsWith('[')) {
    const rows = JSON.parse(raw);
    if (!Array.isArray(rows)) throw new Error('JSON must be an array of day records.');
    return rows;
  }
  const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const header = lines[0].split(',').map(h => h.trim());
  const missing = ['day', ...IMPORT_REQUIRED].filter(k => !header.includes(k));
  if (missing.length) throw new Error(`missing columns: ${missing.join(', ')}`);
  return lines.slice(1).map((line, i) => {
    const cells = line.split(',').map(x => x.trim());
    const row = Object.fromEntries(header.map((h, j) => [h, cells[j]]));
    const rec = { day: row.day || `Day ${i + 1}` };
    if (row.date) rec.date = row.date;
    for (const k of [...IMPORT_REQUIRED, 'fatBurn']) {
      if (k === 'fatBurn' && (row[k] === undefined || row[k] === '')) continue;
      const v = Number(row[k]);
      if (row[k] === undefined || row[k] === '' || Number.isNaN(v)) throw new Error(`row ${i + 1}: ${k} must be a number`);
      rec[k] = v;
    }
    return rec;
  });
}

$('fillSampleImportBtn').addEventListener('click', () => {
  $('importTextarea').value = 'day,steps,rhr,sleepScore,deepMin,remMin,cardio,peak,readiness,fatigue,soreness,stress\nDay 1,9100,59,81,85,92,22,6,8,2,2,2\nDay 2,10400,58,83,88,95,24,7,8,2,2,2\nDay 3,6200,64,61,50,66,8,0,4,4,4,4\nDay 4,7900,60,76,70,80,14,3,7,3,2,2';
});
$('confirmImportBtn').addEventListener('click', async () => {
  const raw = $('importTextarea').value.trim();
  if (!raw) return;
  let parsedDays;
  try {
    parsedDays = parseImport(raw);
  } catch (err) {
    showToast(`Import failed: ${err.message}`);
    return;
  }
  if (!parsedDays.length) { showToast('Import failed: no rows found.'); return; }
  if (!confirm(`Replace this profile's ${getPersona().days.length} days with ${parsedDays.length} imported days?`)) return;
  try {
    const p = getPersona();
    const data = await api(`/api/import/${encodeURIComponent(p.id.toLowerCase())}`, { method: 'POST', body: { days: parsedDays } });
    p.days = data.days;
    p.sessions = data.sessions;
    p.baseline = data.baseline;
    state.livePrediction = undefined;
    renderAll();
    syncPredictionFromBackend();
    showToast(`Imported ${p.days.length} days. See them on the Data page.`);
  } catch (err) {
    showToast(`Import failed: ${err.message}`);
  }
});

$('exportReportBtn').addEventListener('click', () => {
  const p = getPersona();
  const blob = new Blob([JSON.stringify({ exportedAt: new Date().toISOString(), participant: p.name, days: p.days, memories: p.memories, sessions: p.sessions }, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = `CareCompanion_${p.id}_Record.json`;
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  showToast('Exported JSON record');
});

$('exportMarkdownBtn').addEventListener('click', () => {
  const p = getPersona();
  const md = [
    `# CareCompanion Longitudinal Record: ${p.name}`,
    `Profile: ${p.role}`,
    `\n## Persistent Agent Memory\n` + p.memories.map(m => `- **[${m.category} | ${m.session}]** ${m.text}`).join('\n'),
    `\n## Session Evolution\n` + p.sessions.map(s => `### ${s.title} (${s.dayLabel})\nAdaptation: ${s.adaptation}`).join('\n\n')
  ].join('\n');
  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = `CareCompanion_${p.id}_Summary.md`;
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  showToast('Exported Markdown summary');
});

window.addEventListener('DOMContentLoaded', () => {
  renderAll();
  loadModelInfo();
  if (ACCOUNT) loadParticipantState(state.activePersonaId);
  else if (window.CCTracker) window.CCTracker.render(null); // signed out: the watch shows sample stats
});
function syncThemeToggle() {
  const dark = document.documentElement.dataset.theme === 'dark';
  const btn = $('themeToggleBtn');
  $('themeToggleLabel').textContent = dark ? 'Light mode' : 'Dark mode';
  btn.setAttribute('aria-label', dark ? 'Light mode' : 'Dark mode');
  btn.setAttribute('aria-pressed', String(dark));
}
$('themeToggleBtn').addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('carecompanion_theme', next); } catch (e) {}
  syncThemeToggle();
  renderAll();
});
syncThemeToggle();

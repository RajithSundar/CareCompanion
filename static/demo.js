// The /demo page: runs three simulated sessions and lays them out side by side with a no-memory answer.
(function () {
  if (document.body.dataset.page !== 'demo') return;

  const READINGS = [
    ['resting_heart_rate', 'Resting heart rate', ' bpm'],
    ['sleep_score', 'Sleep score', ''],
    ['readiness', 'Feels ready', '/10'],
    ['steps', 'Steps', '']
  ];
  const num = v => (v === null || v === undefined ? null : Number(v));
  const show = (v, unit) => (v === null ? 'not recorded' : `${Math.round(v).toLocaleString()}${unit}`);

  function reading(key, label, unit, now, prev) {
    const v = num(now[key]);
    const p = prev ? num(prev[key]) : null;
    let delta = '';
    if (v !== null && p !== null && Math.round(v) !== Math.round(p)) {
      const d = Math.round(v - p);
      delta = `<span class="demo-delta">${d > 0 ? '+' : ''}${d.toLocaleString()} since last</span>`;
    }
    return `<div class="demo-reading"><span class="demo-reading-label">${esc(label)}</span><span class="demo-reading-value">${esc(show(v, unit))}</span>${delta}</div>`;
  }

  function chips(list, empty) {
    if (!list.length) return `<span class="text-muted">${esc(empty)}</span>`;
    return list.map(m => `<span class="demo-chip" data-cat="${esc(m.category)}">${esc(({ Goal: 'Goal', Constraint: 'Injury or limit', Baseline: 'About them', Adaptation: 'Rule' }[m.category] || m.category))}: ${esc(m.text)}</span>`).join('');
  }

  function render(data) {
    document.getElementById('demoMeta').textContent =
      `${data.name}. Replies written by ${data.provider}. The demo starts with no saved facts and builds them as it goes. Nothing is saved to your profile.`;
    const list = document.getElementById('demoSessions');
    list.innerHTML = data.sessions.map((s, i) => {
      const prev = i > 0 ? data.sessions[i - 1].snapshot : null;
      const f = s.forecast ? s.forecast.predicted_recovery_state : 'Unavailable';
      return `
      <li class="demo-session">
        <div class="demo-rail"><span class="demo-num">${i + 1}</span></div>
        <article class="panels demo-card">
          <header class="demo-card-head">
            <h2 class="demo-card-title">${esc(s.label)}<span class="demo-day">${esc(s.day)}</span></h2>
            <span class="demo-forecast" data-state="${esc(f)}">Tomorrow: ${esc({ 'Ready': 'Ready to train', 'Moderate': 'Take it easy', 'Deload Needed': 'Rest day' }[f] || f)}</span>
          </header>
          <div class="demo-readings">${READINGS.map(([k, l, u]) => reading(k, l, u, s.snapshot, prev)).join('')}</div>
          <blockquote class="demo-quote"><span class="demo-quote-label">They said</span>${esc(s.message)}</blockquote>
          <div class="demo-replies">
            <div class="demo-reply is-memory">
              <div class="demo-reply-label">Our coach (remembers them)</div>
              ${renderAgentSteps(s.steps)}
              <div class="demo-reply-text">${formatMarkdownSimple(s.advice)}</div>
              ${s.plan ? `<div class="demo-plan"><span class="demo-memory-label">Plan it set</span><ul>${s.plan.map(p => `<li>${esc(p.text)}</li>`).join('')}</ul></div>` : ''}
            </div>
            <div class="demo-reply">
              <div class="demo-reply-label">A regular AI (forgets)</div>
              <div class="demo-reply-text">${formatMarkdownSimple(s.stateless_advice)}</div>
            </div>
          </div>
          <footer class="demo-memory">
            <div><span class="demo-memory-label">Remembered going in</span><div class="demo-chips">${chips(s.memories_used, 'Nothing yet: first session.')}</div></div>
            <div><span class="demo-memory-label">Saved from this session</span><div class="demo-chips">${chips(s.new_memories, 'No new facts.')}</div></div>
          </footer>
        </article>
      </li>`;
    }).join('');
  }

  async function loadProfiles() {
    try {
      const data = await api('/api/demo/profiles');
      const sel = document.getElementById('demoProfile');
      const personas = data.profiles.filter(p => p.kind === 'persona');
      const real = data.profiles.filter(p => p.kind === 'pmdata');
      sel.innerHTML =
        `<optgroup label="Fictional personas">${personas.map(p => `<option value="${esc(p.id)}">${esc(p.label.replace(' (fictional persona)', ''))}</option>`).join('')}</optgroup>` +
        (real.length ? `<optgroup label="Real PMData participants">${real.map(p => `<option value="${esc(p.id)}">${esc(p.label.replace('PMData participant ', ''))}</option>`).join('')}</optgroup>` : '');
      sel.value = state.activePersonaId.toLowerCase();
    } catch (e) {
      showToast(`Could not load profiles: ${e.message}`);
    }
  }

  async function run(e) {
    e.preventDefault();
    const btn = document.getElementById('demoRunBtn');
    const profile = document.getElementById('demoProfile').value;
    btn.disabled = true;
    btn.textContent = 'Playing…';
    document.getElementById('demoMeta').textContent = 'Playing the three check-ins in order. This takes up to a minute.';
    document.getElementById('demoSessions').innerHTML = '';
    try {
      render(await api('/api/demo/run', { method: 'POST', body: { profile } }));
    } catch (err) {
      document.getElementById('demoMeta').textContent = `The demo did not finish: ${err.message}`;
    } finally {
      btn.disabled = false;
      btn.textContent = 'Play again';
    }
  }

  window.addEventListener('DOMContentLoaded', () => {
    loadProfiles();
    document.getElementById('demoForm').addEventListener('submit', run);
  });
})();

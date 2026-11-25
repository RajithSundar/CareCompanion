// The 3D tracker in the hero. Turns with the cursor over the hero, spins when dragged,
// and responds to the arrow keys. app.js calls CCTracker.render() with the active
// profile's stats; until a profile loads it shows a labelled random sample.
(function () {
  const stage = document.getElementById('trackerStage');
  const hero = document.getElementById('hero');
  if (!stage || !hero) return;

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  const BASE = { rx: -12, ry: -30 };
  let tiltX = 0;     // from cursor position over the hero
  let tiltY = 0;
  let spin = 0;      // from dragging and the arrow keys
  let pitch = 0;
  let frame = 0;
  let drag = null;

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  function apply() {
    frame = 0;
    const rx = clamp(BASE.rx + pitch + tiltY, -45, 30);
    const ry = BASE.ry + spin + tiltX;
    stage.style.setProperty('--rx', `${rx.toFixed(2)}deg`);
    stage.style.setProperty('--ry', `${ry.toFixed(2)}deg`);
    stage.style.setProperty('--sx', (Math.sin((ry * Math.PI) / 180) * 45).toFixed(2));
  }
  function schedule() { if (!frame) frame = requestAnimationFrame(apply); }

  // Cursor over the hero: a gentle tilt toward the pointer. Skipped under reduced motion.
  hero.addEventListener('pointermove', (e) => {
    if (reduce.matches || drag || e.pointerType === 'touch') return;
    const r = hero.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5;
    const py = (e.clientY - r.top) / r.height - 0.5;
    tiltX = px * 40;
    tiltY = -py * 22;
    stage.classList.add('is-live');
    schedule();
  }, { passive: true });
  hero.addEventListener('pointerleave', () => {
    if (drag) return;
    tiltX = 0; tiltY = 0;
    stage.classList.remove('is-live');
    schedule();
  });

  // Drag to spin. Direct manipulation, so it also works under reduced motion.
  stage.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    drag = { x: e.clientX, y: e.clientY, spin, pitch };
    stage.setPointerCapture(e.pointerId);
    stage.classList.add('is-dragging', 'is-live');
  });
  stage.addEventListener('pointermove', (e) => {
    if (!drag) return;
    spin = drag.spin + (e.clientX - drag.x) * 0.6;
    pitch = clamp(drag.pitch - (e.clientY - drag.y) * 0.3, -30, 30);
    schedule();
  });
  const endDrag = (e) => {
    if (!drag) return;
    drag = null;
    if (stage.hasPointerCapture(e.pointerId)) stage.releasePointerCapture(e.pointerId);
    stage.classList.remove('is-dragging', 'is-live');
  };
  stage.addEventListener('pointerup', endDrag);
  stage.addEventListener('pointercancel', endDrag);

  stage.addEventListener('keydown', (e) => {
    const step = { ArrowLeft: [-20, 0], ArrowRight: [20, 0], ArrowUp: [0, 8], ArrowDown: [0, -8] }[e.key];
    if (step) {
      spin += step[0];
      pitch = clamp(pitch + step[1], -30, 30);
    } else if (e.key === 'Home') {
      spin = 0; pitch = 0;
    } else {
      return;
    }
    e.preventDefault();
    stage.classList.remove('is-live');
    schedule();
  });

  const el = id => document.getElementById(id);
  const fmt = n => Number(n).toLocaleString();

  function randomStats() {
    const r = (lo, hi) => Math.round(lo + Math.random() * (hi - lo));
    return {
      sample: true,
      who: 'Sample',
      day: 'Random',
      steps: r(4200, 12400),
      rhr: r(52, 72),
      sleep: r(56, 90),
      readiness: r(3, 9),
      forecast: ['Ready', 'Moderate', 'Deload Needed'][r(0, 2)]
    };
  }

  function render(d) {
    const data = d || randomStats();
    el('trkWho').textContent = data.who;
    el('trkDay').textContent = data.day;
    el('trkSteps').textContent = fmt(data.steps);
    el('trkRhr').textContent = data.rhr;
    el('trkSleep').textContent = data.sleep;
    el('trkReadyVal').textContent = `${data.readiness}/10`;
    const bar = el('trkReadyBar');
    if (bar.children.length !== 10) bar.innerHTML = '<span></span>'.repeat(10);
    [...bar.children].forEach((s, i) => s.classList.toggle('on', i < Number(data.readiness)));
    el('trkForecast').dataset.state = data.forecast || 'none';
    el('trkForecastVal').textContent = data.forecast ? data.forecast.replace(' Needed', '') : '--';
    el('trackerSummary').textContent = data.sample
      ? 'Random sample stats. No profile is loaded.'
      : `${data.who}, ${data.day}: ${fmt(data.steps)} steps, resting heart rate ${data.rhr}, sleep score ${data.sleep}, readiness ${data.readiness} of 10.${data.forecast ? ` Tomorrow: ${data.forecast}.` : ''}`;
  }

  let received = false;
  window.CCTracker = {
    render(d) {
      if (d) received = true;
      if (d || !received) render(d);
    }
  };
  // No profile within a few seconds (server down, offline): show a labelled random sample.
  setTimeout(() => { if (!received) render(null); }, 3000);
  apply();
})();

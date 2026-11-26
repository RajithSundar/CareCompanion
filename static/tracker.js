// The 3D watch in the hero. Tilts toward the cursor while it is over the watch, spins when dragged,
// and responds to the arrow keys. app.js calls CCTracker.render() with the active
// profile's stats; until a profile loads it shows a labelled random sample.
(function () {
  const stage = document.getElementById('trackerStage');
  if (!stage) return;

  const tracker = document.getElementById('tracker');
  const glassSheen = document.getElementById('trkGlassSheen');
  const rimSheen = document.getElementById('trkRimSheen');
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');

  // Performance: only transforms change per frame, written straight onto the elements that move
  // (no CSS variables on the stage, which would restyle every element in the watch), from one
  // requestAnimationFrame loop that stops as soon as the watch settles.
  const REST = { rx: -6, ry: -24 };   // resting pose, turned toward the page copy
  const FOLLOW = 0.22;                // easing while the cursor is on the watch
  const GLIDE = 0.075;                // easing on the way back to rest
  let tiltX = 0, tiltY = 0;           // from the cursor
  let spin = 0, pitch = 0;            // from dragging and the arrow keys
  let cur = { ...REST };
  let ease = GLIDE;
  let frame = 0;
  let drag = null;
  let rect = null;                    // stage bounds, cached so pointermove never forces layout

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const target = () => ({ rx: clamp(REST.rx + pitch + tiltY, -45, 30), ry: REST.ry + spin + tiltX });

  function draw() {
    tracker.style.transform = `rotateX(${cur.rx.toFixed(2)}deg) rotateY(${cur.ry.toFixed(2)}deg)`;
    const sx = Math.sin((cur.ry * Math.PI) / 180) * 45;
    glassSheen.style.transform = `translateX(${sx.toFixed(1)}%)`;
    rimSheen.style.transform = `translateX(${(sx * 1.4).toFixed(1)}%)`;
  }

  function step() {
    frame = 0;
    const t = target();
    const k = drag || reduce.matches ? 1 : ease;
    cur.rx += (t.rx - cur.rx) * k;
    cur.ry += (t.ry - cur.ry) * k;
    if (Math.abs(t.rx - cur.rx) < 0.02 && Math.abs(t.ry - cur.ry) < 0.02) cur = t; // settled: loop ends
    else frame = requestAnimationFrame(step);
    draw();
  }
  function schedule(k) { ease = k; if (!frame) frame = requestAnimationFrame(step); }

  const measure = () => { rect = stage.getBoundingClientRect(); };
  stage.addEventListener('pointerenter', measure);
  window.addEventListener('resize', () => { rect = null; }, { passive: true });
  window.addEventListener('scroll', () => { rect = null; }, { passive: true });

  // Cursor over the watch: it tilts toward the pointer. Skipped under reduced motion.
  stage.addEventListener('pointermove', (e) => {
    if (drag) {
      spin = drag.spin + (e.clientX - drag.x) * 0.6;
      pitch = clamp(drag.pitch - (e.clientY - drag.y) * 0.3, -30, 30);
      schedule(1);
      return;
    }
    if (reduce.matches || e.pointerType === 'touch') return;
    if (!rect) measure();
    tiltX = ((e.clientX - rect.left) / rect.width - 0.5) * 50;
    tiltY = -((e.clientY - rect.top) / rect.height - 0.5) * 30;
    schedule(FOLLOW);
  }, { passive: true });

  // Cursor gone: glide back to the resting pose.
  function rest() {
    tiltX = 0; tiltY = 0; spin = 0; pitch = 0;
    stage.classList.remove('is-dragging');
    schedule(GLIDE);
  }
  stage.addEventListener('pointerleave', () => { if (!drag) rest(); });
  stage.addEventListener('blur', () => { if (!drag && !stage.matches(':hover')) rest(); });

  // Drag to spin. Direct manipulation, so it also works under reduced motion.
  stage.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    drag = { x: e.clientX, y: e.clientY, spin, pitch };
    stage.setPointerCapture(e.pointerId);
    stage.classList.add('is-dragging');
  });
  const endDrag = (e) => {
    if (!drag) return;
    drag = null;
    if (stage.hasPointerCapture(e.pointerId)) stage.releasePointerCapture(e.pointerId);
    stage.classList.remove('is-dragging');
    measure();
    const inside = e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom;
    if (!inside || e.pointerType === 'touch') rest(); // released off the watch (or a finger lifted)
  };
  stage.addEventListener('pointerup', endDrag);
  stage.addEventListener('pointercancel', endDrag);

  stage.addEventListener('keydown', (e) => {
    const k = { ArrowLeft: [-20, 0], ArrowRight: [20, 0], ArrowUp: [0, 8], ArrowDown: [0, -8] }[e.key];
    if (k) {
      spin += k[0];
      pitch = clamp(pitch + k[1], -30, 30);
    } else if (e.key === 'Home' || e.key === 'Escape') {
      spin = 0; pitch = 0;
    } else {
      return;
    }
    e.preventDefault();
    schedule(FOLLOW);
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

  const PLAIN = { 'Ready': 'ready to train', 'Moderate': 'take it easy', 'Deload Needed': 'rest day' };

  function render(d) {
    if (d && d.empty) {
      ['trkSteps', 'trkRhr', 'trkSleep', 'trkReadyVal', 'trkForecastVal'].forEach(id => { el(id).textContent = '--'; });
      el('trkRing').setAttribute('stroke-dasharray', '0 100');
      el('trkForecast').dataset.state = 'none';
      el('trackerSummary').textContent = `${d.who}, no days logged yet. Log a check-in to fill the watch.`;
      return;
    }
    const data = d || randomStats();
    el('trkSteps').textContent = fmt(data.steps);
    el('trkRhr').textContent = data.rhr;
    el('trkSleep').textContent = data.sleep;
    el('trkReadyVal').textContent = data.readiness;
    const goal = 10000;
    el('trkRing').setAttribute('stroke-dasharray', `${Math.min(100, (Number(data.steps) / goal) * 100).toFixed(1)} 100`);
    el('trkForecast').dataset.state = data.forecast || 'none';
    el('trkForecastVal').textContent = data.forecast ? ({ 'Ready': 'Train', 'Moderate': 'Easy', 'Deload Needed': 'Rest' }[data.forecast] || data.forecast) : '--';
    el('trackerSummary').textContent = data.sample
      ? 'Example numbers. Sign in to see yours.'
      : `${data.who}, ${data.day}: ${fmt(data.steps)} steps, resting heart rate ${data.rhr}, sleep score ${data.sleep}, readiness ${data.readiness} of 10.${data.forecast ? ` Tomorrow: ${PLAIN[data.forecast] || data.forecast}.` : ''}`;
  }

  // The watch shows the visitor's own clock.
  function tick() {
    const now = new Date();
    el('trkTime').textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    el('trkDate').textContent = now.toLocaleDateString([], { weekday: 'short', day: 'numeric' });
  }
  tick();
  setInterval(tick, 30000);

  let received = false;
  window.CCTracker = {
    render(d) {
      if (d) received = true;
      if (d || !received) render(d);
    }
  };
  // No profile within a few seconds (server down, offline): show a labelled random sample.
  setTimeout(() => { if (!received) render(null); }, 3000);
  draw();
})();

// Sign-in and sign-up pages. The server sets an HttpOnly cookie; this script never sees the token.
(function () {
  const form = document.getElementById('loginForm');
  if (!form) return;
  const user = document.getElementById('loginUsername');
  const pass = document.getElementById('loginPassword');
  const err = document.getElementById('loginError');
  const btn = document.getElementById('loginSubmit');

  function nextPath() {
    const n = new URLSearchParams(location.search).get('next') || '/';
    return n.startsWith('/') && !n.startsWith('//') && !n.includes('\\') ? n : '/';
  }

  function showError(msg) {
    err.textContent = msg;
    err.hidden = !msg;
  }

  async function signIn(username, password) {
    showError('');
    if (!username || !password) { showError('Enter a username and password.'); return; }
    btn.disabled = true;
    btn.textContent = 'Signing in…';
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Sign-in failed. Try again.');
      location.replace(nextPath());
    } catch (e) {
      showError(e.message === 'Failed to fetch' ? 'Could not reach the server.' : e.message);
      btn.disabled = false;
      btn.textContent = 'Sign in';
      pass.focus();
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    signIn(user.value.trim(), pass.value);
  });

  const pw = document.getElementById('demoPassword');
  document.querySelectorAll('.login-account').forEach((b) => {
    b.addEventListener('click', () => {
      user.value = b.dataset.username;
      if (pw) { pass.value = pw.textContent; signIn(user.value, pass.value); }
      else pass.focus();
    });
  });

  user.focus({ preventScroll: true });
})();

// Sign-up page. Same cookie as sign-in; the new account opens on its Check-in page if it started empty.
(function () {
  const form = document.getElementById('signupForm');
  if (!form) return;
  const err = document.getElementById('signupError');
  const btn = document.getElementById('signupSubmit');
  const sampleField = document.getElementById('signupSampleField');
  const sample = document.getElementById('signupSample');
  const val = id => document.getElementById(id).value;

  function startChoice() {
    return form.querySelector('input[name="start"]:checked').value;
  }
  form.querySelectorAll('input[name="start"]').forEach(r => r.addEventListener('change', () => {
    sampleField.hidden = startChoice() !== 'sample';
  }));
  if (sample && sample.options.length) sample.selectedIndex = Math.floor(Math.random() * sample.options.length);
  if (!sample || !sample.options.length) {
    const opt = form.querySelector('input[name="start"][value="sample"]');
    if (opt) opt.closest('label').hidden = true;
  }

  function showError(msg) {
    err.textContent = msg;
    err.hidden = !msg;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    showError('');
    const name = val('signupName').trim();
    const username = val('signupUsername').trim().toLowerCase();
    const password = val('signupPassword');
    if (!name) { showError('Enter your name.'); return; }
    if (!/^[a-z0-9_]{3,20}$/.test(username)) { showError('Usernames are 3 to 20 characters: lowercase letters, digits or _.'); return; }
    if (password.length < 8) { showError('Use a password of at least 8 characters.'); return; }
    const start = startChoice() === 'sample' ? sample.value : 'fresh';
    btn.disabled = true;
    btn.textContent = 'Creating account…';
    try {
      const res = await fetch('/api/auth/signup', {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, username, password, start })
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        let msg = data.detail;
        if (Array.isArray(msg)) msg = msg.map(d => d.msg).join('; ');
        throw new Error(typeof msg === 'string' ? msg : 'Could not create the account. Try again.');
      }
      location.replace(start === 'fresh' ? '/checkin' : '/coach');
    } catch (ex) {
      showError(ex.message === 'Failed to fetch' ? 'Could not reach the server.' : ex.message);
      btn.disabled = false;
      btn.textContent = 'Create account';
    }
  });

  document.getElementById('signupName').focus({ preventScroll: true });
})();

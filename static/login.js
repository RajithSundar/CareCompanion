// Sign-in page. The server sets an HttpOnly cookie; this script never sees the token.
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

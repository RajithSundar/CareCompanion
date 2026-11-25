// Runs before first paint so the saved theme does not flash.
// ?theme=dark or ?theme=light in the URL overrides the saved choice for that visit.
(function applySavedTheme() {
  let saved = null;
  try { saved = localStorage.getItem('carecompanion_theme'); } catch (e) {}
  const forced = new URLSearchParams(window.location.search).get('theme');
  if (forced === 'dark' || forced === 'light') saved = forced;
  document.documentElement.dataset.theme = saved === 'dark' ? 'dark' : 'light';
})();

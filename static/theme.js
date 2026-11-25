// Runs before first paint so the saved theme does not flash.
(function applySavedTheme() {
  let saved = null;
  try { saved = localStorage.getItem('carecompanion_theme'); } catch (e) {}
  document.documentElement.dataset.theme = saved === 'dark' ? 'dark' : 'light';
})();

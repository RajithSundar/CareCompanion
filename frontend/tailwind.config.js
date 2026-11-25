module.exports = {
  content: ["../static/pages/*.html", "../static/partials/*.html", "../static/app.js", "../static/tracker.js", "../static/demo.js"],
  future: { hoverOnlyWhenSupported: true },
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Host Grotesk"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"Atkinson Hyperlegible Mono"', 'ui-monospace', 'monospace']
      },
      colors: {
        'paper': 'var(--color-paper)',
        'paper-2': 'var(--color-paper-2)',
        'paper-3': 'var(--color-paper-3)',
        'ink': 'var(--color-ink)',
        'ink-2': 'var(--color-ink-2)',
        'muted': 'var(--color-muted)',
        'rule': 'var(--color-rule)',
        'rule-2': 'var(--color-rule-2)',
        'accent': 'var(--color-accent)',
        'accent-ink': 'var(--color-accent-ink)',
        'accent-soft': 'var(--color-accent-soft)',
        'cool': 'var(--color-cool)',
        'cool-soft': 'var(--color-cool-soft)',
        'ok': 'var(--color-ok)',
        'ok-soft': 'var(--color-ok-soft)',
        'bad': 'var(--color-bad)',
        'bad-soft': 'var(--color-bad-soft)'
      }
    }
  }
};

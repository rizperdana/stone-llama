/* stone-llama site — no build step, no dependencies.
   Progressive enhancement only: theme toggle + copy button.
   The page is complete without this file (theme follows the OS,
   copy button and toggle are hidden until html.js adds them). */
(function () {
  'use strict';

  var root = document.documentElement;

  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var current = root.dataset.theme ||
        (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
      var next = current === 'light' ? 'dark' : 'light';
      root.dataset.theme = next;
      try { localStorage.setItem('theme', next); } catch (e) {}
    });
  }

  document.querySelectorAll('[data-copy]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var src = document.getElementById(btn.dataset.copy);
      if (!src || !navigator.clipboard) return;
      navigator.clipboard.writeText(src.textContent.trim()).then(function () {
        var old = btn.textContent;
        btn.textContent = 'copied';
        setTimeout(function () { btn.textContent = old; }, 1600);
      }, function () {});
    });
  });
})();

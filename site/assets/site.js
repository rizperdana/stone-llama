/* stone-llama site — no build step, no dependencies.
   Progressive enhancement only: the copy button.
   The page is complete without this file (the copy button is
   hidden until html.js adds it). */
(function () {
  'use strict';

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

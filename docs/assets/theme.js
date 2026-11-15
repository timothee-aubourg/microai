/* Applies the reader's light or dark choice before the page is drawn. The switch is in the top bar (micro.js). */
(function () {
  try { var t = localStorage.getItem('microai.theme'); if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t); } catch (e) { /* no storage */ }
})();

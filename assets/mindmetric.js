/* Rotación automática de vistas del demo MindMetric (pages/mindmetric.py).
 *
 * Solo actúa en la página /mindmetric y solo si el equipo la activa desde el
 * panel de Operación. Con la rotación activada, si nadie toca la pantalla
 * durante IDLE_MS, cambia de vista cada ROTATE_MS. Cualquier interacción real
 * la detiene; los clics que hace el propio script no cuentan como
 * interacción (no son isTrusted). Nunca rota con el panel de Operación o una
 * demostración abiertos.
 */
(function () {
  'use strict';
  var IDLE_MS = 45000;
  var ROTATE_MS = 20000;
  var ORDER = ['persona', 'empresa', 'medimos'];
  var lastInteraction = Date.now();
  var lastRotation = 0;

  function markInteraction(event) {
    if (event.isTrusted) lastInteraction = Date.now();
  }
  ['pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart'].forEach(function (type) {
    document.addEventListener(type, markInteraction, true);
  });

  document.addEventListener('click', function (event) {
    var button = event.target.closest && event.target.closest('#mm-autoplay');
    var root = document.querySelector('.mm-root');
    if (!button || !root) return;
    var on = root.dataset.autoplay !== 'on';
    root.dataset.autoplay = on ? 'on' : 'off';
    button.textContent = on ? button.dataset.labelOff : button.dataset.labelOn;
    button.classList.toggle('is-active', on);
  });

  setInterval(function () {
    var root = document.querySelector('.mm-root');
    if (!root || root.dataset.autoplay !== 'on') return;
    var now = Date.now();
    if (now - lastInteraction < IDLE_MS || now - lastRotation < ROTATE_MS) return;
    if (document.querySelector('#mm-op:not(.is-closed)')) return;
    if (document.querySelector('#mm-demo:not(.is-hidden)')) return;
    var current = ORDER.findIndex(function (key) {
      var b = document.getElementById('mm-aud-' + key);
      return b && b.classList.contains('is-active');
    });
    var next = document.getElementById('mm-aud-' + ORDER[(current + 1) % ORDER.length]);
    if (next) {
      next.click();
      lastRotation = now;
    }
  }, 1000);
})();

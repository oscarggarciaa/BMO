// Cara de BMO: escucha los cambios de expresion por SSE y pinta la cara.
// Cada expresion es una carpeta de frames. Si tiene un solo frame, la cara
// queda fija; si tiene varios, los ciclamos para animarla (la boca al hablar,
// el parpadeo al pensar). Si no hay frames, mostramos el nombre como fallback.

(function () {
  "use strict";

  var img = document.getElementById("face-img");
  var label = document.getElementById("face-label");

  var timer = null; // intervalo de animacion activo
  var currentExpr = ""; // expresion que se esta mostrando

  function stopAnimation() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  function showLabel(expr) {
    img.style.display = "none";
    label.textContent = expr;
    label.style.display = "block";
  }

  function showImage() {
    img.style.display = "block";
    label.style.display = "none";
  }

  // Precarga los frames para que el primer ciclo no parpadee.
  function preload(frames) {
    frames.forEach(function (src) {
      var pre = new Image();
      pre.src = src;
    });
  }

  function animate(frames, fps) {
    stopAnimation();
    if (!frames || frames.length === 0) {
      showLabel(currentExpr);
      return;
    }
    preload(frames);

    img.onload = showImage;
    img.onerror = function () {
      showLabel(currentExpr);
    };

    var i = 0;
    function draw() {
      img.src = frames[i];
      i = (i + 1) % frames.length;
    }
    draw();

    // Un solo frame = cara fija. Varios = animacion en loop.
    if (frames.length > 1) {
      var period = Math.max(1, Math.round(1000 / (fps || 4)));
      timer = setInterval(draw, period);
    }
  }

  function setExpression(expr) {
    currentExpr = expr;
    label.textContent = expr;
    fetch("/faces/" + encodeURIComponent(expr))
      .then(function (r) {
        return r.ok ? r.json() : { frames: [], fps: 4 };
      })
      .then(function (data) {
        // Anti race-condition: si mientras pediamos el manifest la cara ya
        // cambio a otra, esta respuesta llego tarde y hay que ignorarla. Sin
        // esto, un fetch lento puede pisar la expresion nueva (p.ej. quedarse
        // pegado en 'talking' en vez de volver a 'listening').
        if (expr !== currentExpr) {
          return;
        }
        animate(data.frames, data.fps);
      })
      .catch(function () {
        if (expr !== currentExpr) {
          return;
        }
        stopAnimation();
        showLabel(expr);
      });
  }

  function connect() {
    var source = new EventSource("/events");
    source.onmessage = function (event) {
      setExpression(event.data.trim());
    };
    source.onerror = function () {
      // El navegador reintenta solo; si se cae del todo, forzamos reconexion.
      source.close();
      setTimeout(connect, 2000);
    };
  }

  connect();
})();

// Caden Hill portfolio. The only shared script: draws the red pencil marks as they come into view.
(function () {
  var marks = document.querySelectorAll('svg.mark');
  if (!marks.length || !('IntersectionObserver' in window)) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  marks.forEach(function (svg) {
    svg.querySelectorAll('path').forEach(function (p) {
      // the svg is stretched to the image, so measure in its own units
      p.style.setProperty('--len', Math.ceil(p.getTotalLength()) + 1);
    });
    svg.classList.add('ready');
    var note = svg.closest('figure') && svg.closest('figure').querySelector('.pencil');
    if (note) note.classList.add('ready');
  });

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      var svg = e.target;
      // a beat of delay so it reads as being drawn, not loaded
      setTimeout(function () {
        svg.classList.add('drawn');
        var note = svg.closest('figure') && svg.closest('figure').querySelector('.pencil');
        if (note) note.classList.add('drawn');
      }, 250);
      io.unobserve(svg);
    });
  }, { threshold: 0.6 });

  marks.forEach(function (svg) { io.observe(svg); });
})();

// Diagram flow lines run while the diagram is on screen (phones have no hover).
(function () {
  var figs = document.querySelectorAll('.fig');
  if (!figs.length || !('IntersectionObserver' in window)) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { e.target.classList.toggle('live', e.isIntersecting); });
  }, { threshold: 0.5 });
  figs.forEach(function (f) { io.observe(f); });
})();

// Pencil strokes under key phrases draw in once they're on screen.
(function () {
  var marks = document.querySelectorAll('.mk');
  if (!marks.length) return;
  if (!('IntersectionObserver' in window)) { marks.forEach(function (m) { m.classList.add('on'); }); return; }
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { setTimeout(function () { e.target.classList.add('on'); }, 300); io.unobserve(e.target); }
    });
  }, { threshold: 1 });
  marks.forEach(function (m) { io.observe(m); });
})();

// Supplier's ask vs what aluminum justifies, using the numbers from the should-cost workbook.
(function () {
  var box = document.getElementById('versus');
  if (!box) return;
  var text = {
    ask: 'A flat 9% on the whole part because aluminum went up about 18%. That takes the price from $38.90 to $42.40.<span class="tag">How the increase usually shows up.</span>',
    fair: 'Only the metal moved, so aluminum justifies about 3.2%, or $40.16. The difference is about $2.24 a part, or roughly $26,847 a year.<span class="tag">What I would counter with.</span>'
  };
  var out = document.getElementById('versus-text');
  box.querySelectorAll('button').forEach(function (b) {
    b.addEventListener('click', function () {
      box.querySelectorAll('button').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
      out.innerHTML = text[b.dataset.k];
    });
  });
})();

// Footer date comes from when the page was last published.
(function () {
  var el = document.getElementById('updated');
  if (!el) return;
  var d = new Date(document.lastModified);
  if (isNaN(d)) return;
  el.textContent = 'Updated ' + d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
})();

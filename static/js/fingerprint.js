(function () {
  const ROOT = 'static/images/fingerprint';
  const VERSION = 'v=1';
  const N_SRC = 1.4;
  const N_MIN = 1.4;
  const N_MAX = 22;

  // Disparity D is in 1/m; kappa in px·m, so CoC = kappa * (D_focus - D) is in pixels.
  const SCENES = [
    { id: 'ub4', kappa: 247.39, dMin: 0.100803, dMax: 0.814407, focus: [0.3, 0.55] },
    { id: 'ub18', kappa: 294.62, dMin: 0.018638, dMax: 0.467059, focus: [0.45, 0.52] },
  ];

  const root = document.getElementById('fp');
  const source = root.querySelector('.fp-source');
  const map = root.querySelector('.fp-map');
  const mapLabel = root.querySelector('.fp-map-label');
  const markers = root.querySelectorAll('.fp-marker');
  const viewers = root.querySelectorAll('.fp-viewer');
  const plot = root.querySelector('.fp-plot');
  const slider = document.getElementById('fp-aperture');
  const sceneGroup = document.getElementById('fp-scenes');
  const out = {
    aperture: document.getElementById('fp-aperture-value'),
    ksrc: document.getElementById('fp-ksrc'),
    ra: document.getElementById('fp-ra'),
    ktgt: document.getElementById('fp-ktgt'),
  };

  const state = { scene: null, disp: null, w: 0, h: 0, focus: [0.5, 0.5], dFocus: 0, n: 4, samples: [] };
  const cache = {};

  const apertureFromSlider = (t) => N_MIN * Math.pow(N_MAX / N_MIN, t);
  const sliderFromAperture = (n) => Math.log(n / N_MIN) / Math.log(N_MAX / N_MIN);
  const fmtN = (n) => (n < 10 ? n.toFixed(1) : n.toFixed(0));

  // Diverging map: near (negative CoC) warm, in-focus white, far (positive CoC) cool.
  const NEAR = [233, 78, 119];
  const FAR = [59, 92, 214];
  function color(v) {
    const t = Math.max(-1, Math.min(1, v));
    const c = t < 0 ? NEAR : FAR;
    const a = Math.pow(Math.abs(t), 0.6);
    return [255 + (c[0] - 255) * a, 255 + (c[1] - 255) * a, 255 + (c[2] - 255) * a];
  }

  function loadDisparity(scene) {
    if (cache[scene.id]) return Promise.resolve(cache[scene.id]);
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = img.width;
        c.height = img.height;
        const ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const px = ctx.getImageData(0, 0, c.width, c.height).data;
        const disp = new Float32Array(c.width * c.height);
        const span = scene.dMax - scene.dMin;
        for (let i = 0; i < disp.length; i++) {
          disp[i] = scene.dMin + ((px[i * 4] << 8) | px[i * 4 + 1]) / 65535 * span;
        }
        cache[scene.id] = { disp, w: c.width, h: c.height };
        resolve(cache[scene.id]);
      };
      img.src = `${ROOT}/${scene.id}/disp.png?${VERSION}`;
    });
  }

  function disparityAt(fx, fy) {
    const { disp, w, h } = state;
    const cx = Math.round(fx * (w - 1));
    const cy = Math.round(fy * (h - 1));
    const vals = [];
    for (let dy = -2; dy <= 2; dy++) {
      for (let dx = -2; dx <= 2; dx++) {
        const x = Math.min(w - 1, Math.max(0, cx + dx));
        const y = Math.min(h - 1, Math.max(0, cy + dy));
        vals.push(disp[y * w + x]);
      }
    }
    vals.sort((a, b) => a - b);
    return vals[12];
  }

  // Fixed per scene so that stopping down visibly fades the map toward white.
  const colorScale = (scene) => 0.35 * scene.kappa * (scene.dMax - scene.dMin);

  function drawMap() {
    const { disp, w, h, scene, dFocus } = state;
    const kTgt = (N_SRC / state.n) * scene.kappa;
    const scale = colorScale(scene);
    const ctx = map.getContext('2d');
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < disp.length; i++) {
      const [r, g, b] = color((kTgt * (dFocus - disp[i])) / scale);
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function drawPlot() {
    const { scene, dFocus, samples } = state;
    const dpr = window.devicePixelRatio || 1;
    const W = plot.clientWidth;
    const H = plot.clientHeight;
    plot.width = W * dpr;
    plot.height = H * dpr;
    const ctx = plot.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const pad = { l: 46, r: 14, t: 14, b: 42 };
    const span = scene.dMax - scene.dMin;
    const yMax = Math.ceil((scene.kappa * span) / 50) * 50;
    const X = (d) => pad.l + ((d - scene.dMin) / span) * (W - pad.l - pad.r);
    const Y = (c) => pad.t + (1 - (c + yMax) / (2 * yMax)) * (H - pad.t - pad.b);
    const kTgt = (N_SRC / state.n) * scene.kappa;
    const dSrc = scene.srcFocus;

    ctx.font = '12px "Google Sans", sans-serif';
    ctx.fillStyle = '#777';
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b);
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(0));
    ctx.lineTo(W - pad.r, Y(0));
    ctx.stroke();

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    [-yMax, -yMax / 2, 0, yMax / 2, yMax].forEach((c) => ctx.fillText(Math.round(c), pad.l - 6, Y(c)));
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const step = span > 0.5 ? 0.2 : 0.1;
    for (let d = Math.ceil(scene.dMin / step) * step; d <= scene.dMax; d += step) {
      ctx.fillText(d.toFixed(1), X(d), H - pad.b + 5);
    }
    ctx.fillText('Disparity D (1/m)', (pad.l + W - pad.r) / 2, H - 16);
    ctx.save();
    ctx.translate(12, (pad.t + H - pad.b) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = 'middle';
    ctx.fillText('Signed CoC (px)', 0, 0);
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b);
    ctx.clip();

    const dot = (x, y, r) => {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fill();
    };
    ctx.fillStyle = 'rgba(0, 0, 0, 0.18)';
    samples.forEach((d) => dot(X(d), Y(scene.kappa * (dSrc - d)), 2));

    const line = (k, d0, style, width, dash) => {
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(X(scene.dMin), Y(k * (d0 - scene.dMin)));
      ctx.lineTo(X(scene.dMax), Y(k * (d0 - scene.dMax)));
      ctx.stroke();
      ctx.setLineDash([]);
    };
    line(scene.kappa, dSrc, '#999', 1.5, [5, 4]);

    samples.forEach((d) => {
      const c = kTgt * (dFocus - d);
      const [r, g, b] = color(c > 0 ? 1 : -1);
      ctx.fillStyle = `rgba(${r | 0}, ${g | 0}, ${b | 0}, 0.8)`;
      dot(X(d), Y(c), 3);
    });
    line(kTgt, dFocus, '#363636', 2, []);

    ctx.strokeStyle = '#363636';
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(X(dFocus), pad.t);
    ctx.lineTo(X(dFocus), H - pad.b);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    ctx.fillStyle = '#363636';
    ctx.textAlign = X(dFocus) > W - 90 ? 'right' : 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('focus', X(dFocus) + (ctx.textAlign === 'left' ? 5 : -5), pad.t + 4);
  }

  function render() {
    const kTgt = (N_SRC / state.n) * state.scene.kappa;
    out.aperture.textContent = `f/${fmtN(state.n)}`;
    out.ksrc.textContent = `${state.scene.kappa.toFixed(1)} px·m`;
    out.ra.textContent = `${N_SRC} / ${fmtN(state.n)} = ${(N_SRC / state.n).toFixed(3)}`;
    out.ktgt.textContent = `${kTgt.toFixed(1)} px·m`;
    mapLabel.textContent = `Target CoC (f/${fmtN(state.n)})`;
    markers.forEach((m) => {
      m.style.left = `${state.focus[0] * 100}%`;
      m.style.top = `${state.focus[1] * 100}%`;
    });
    drawMap();
    drawPlot();
  }

  function setFocus(fx, fy) {
    state.focus = [fx, fy];
    state.dFocus = disparityAt(fx, fy);
    render();
  }

  function pickSamples() {
    const { disp } = state;
    const samples = [];
    let seed = 7;
    for (let i = 0; i < 900; i++) {
      seed = (seed * 16807) % 2147483647;
      samples.push(disp[seed % disp.length]);
    }
    state.samples = samples;
  }

  function selectScene(scene) {
    source.src = `${ROOT}/${scene.id}/source.jpg?${VERSION}`;
    sceneGroup.querySelectorAll('[data-scene]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.scene === scene.id));
    loadDisparity(scene).then((data) => {
      Object.assign(state, data, { scene });
      map.width = data.w;
      map.height = data.h;
      pickSamples();
      setFocus(scene.focus[0], scene.focus[1]);
    });
  }

  SCENES.forEach((scene) => {
    scene.srcFocus = { ub4: 100 / 192, ub18: 100 / 232 }[scene.id];
    const thumb = document.createElement('button');
    thumb.className = 'demo-thumb';
    thumb.dataset.scene = scene.id;
    thumb.innerHTML = `<img src="${ROOT}/${scene.id}/source.jpg?${VERSION}" alt="UnrealBokeh scene" loading="lazy">`;
    thumb.onclick = () => selectScene(scene);
    sceneGroup.appendChild(thumb);
  });

  viewers.forEach((viewer) => {
    const pick = (e) => {
      const rect = viewer.getBoundingClientRect();
      const fx = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      const fy = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
      if (state.disp) setFocus(fx, fy);
    };
    viewer.addEventListener('pointerdown', (e) => {
      viewer.setPointerCapture(e.pointerId);
      pick(e);
    });
    viewer.addEventListener('pointermove', (e) => {
      if (viewer.hasPointerCapture(e.pointerId)) pick(e);
    });
  });

  slider.value = sliderFromAperture(state.n);
  slider.addEventListener('input', () => {
    state.n = apertureFromSlider(parseFloat(slider.value));
    if (state.disp) render();
  });

  window.addEventListener('resize', () => { if (state.disp) drawPlot(); });

  selectScene(SCENES[0]);
})();

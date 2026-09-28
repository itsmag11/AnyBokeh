(function () {
  const ROOT = 'static/images/demo';
  const VERSION = 'v=5';
  const STOPS = ['2', '2.2', '2.5', '2.8', '3.2', '3.5', '4', '4.5', '5', '5.6', '6.3',
    '7.1', '8', '9', '10', '11', '13', '14', '16', '18', '20'];

  // `source` is the f-number of the input photo; `id` is the image folder name.
  const SCENES = [
    { id: '63_f13', source: 'f/13', focus: ['focus1', 'focus2'] },
    { id: '119_f22', source: 'f/22', focus: ['focus1', 'focus2', 'focus3'] },
    { id: '51_f2.0', source: 'f/2.0', focus: ['focus1'] },
    { id: '127_f5.0', source: 'f/5.0', focus: ['focus1', 'focus2', 'focus3'] },
    { id: '72_f22', source: 'f/22', focus: ['focus1'] },
    { id: '165_f16', source: 'f/16', focus: ['focus3'] },
    { id: '120_f5.0', source: 'f/5.0', focus: ['focus2', 'focus3'] },
    { id: '168_f6.3', source: 'f/6.3', focus: ['focus3'] },
    { id: '205_f4.5', source: 'f/4.5', focus: ['focus1'] },
    { id: '85_f13', source: 'f/13', focus: ['focus1'] },
  ];

  const REVIEW = new URLSearchParams(location.search).has('review');

  const state = { scene: SCENES[0], focus: SCENES[0].focus[0], stop: 0 };

  const inputImg = document.getElementById('demo-input');
  const inputLabel = document.getElementById('demo-input-label');
  const img = document.getElementById('demo-image');
  const label = document.getElementById('demo-label');
  const slider = document.getElementById('demo-aperture');
  const sliderValue = document.getElementById('demo-aperture-value');
  const focusGroup = document.getElementById('demo-focus');
  const sceneGroup = document.getElementById('demo-scenes');

  const src = (scene, focus, key) => `${ROOT}/${scene.id}/${focus}_${key}.jpg?${VERSION}`;
  const inputSrc = () => src(state.scene, state.focus, 'input');
  const frameSrc = (stop) => src(state.scene, state.focus, 'f' + STOPS[stop]);

  // A frame counts as ready only once decoded, so assigning it to <img> never stalls on decode.
  const images = {};
  const ready = new Set();
  function load(url) {
    if (images[url]) return;
    const im = new Image();
    images[url] = im;
    im.src = url;
    const done = im.decode ? im.decode() : new Promise((res, rej) => { im.onload = res; im.onerror = rej; });
    done.then(() => { ready.add(url); paint(); }).catch(() => {});
  }

  // Input first, then frames nearest to the current stop, so the slider position fills in first.
  function preload() {
    load(inputSrc());
    STOPS.map((_, i) => i)
      .sort((a, b) => Math.abs(a - state.stop) - Math.abs(b - state.stop))
      .forEach((i) => load(frameSrc(i)));
  }

  function nearestReadyStop() {
    for (let d = 0; d < STOPS.length; d++) {
      for (const s of [state.stop - d, state.stop + d]) {
        if (s >= 0 && s < STOPS.length && ready.has(frameSrc(s))) return s;
      }
    }
    return -1;
  }

  function show(el, url) {
    if (el.getAttribute('src') !== url) el.src = url;
    el.classList.remove('is-fading');
  }

  // While the target frame is loading, show the closest loaded stop of the same view; until
  // anything of the new view is loaded, keep the previous image dimmed.
  function paint() {
    const input = inputSrc();
    if (ready.has(input)) show(inputImg, input);
    else inputImg.classList.add('is-fading');

    const s = nearestReadyStop();
    if (s >= 0) show(img, frameSrc(s));
    else img.classList.add('is-fading');
  }

  function render() {
    const n = 'f/' + STOPS[state.stop];
    sliderValue.textContent = n;
    label.textContent = 'AnyBokeh · ' + n;
    inputLabel.textContent = `Input (${state.scene.source})`;
    paint();
  }

  function selectView(scene, focus) {
    state.scene = scene;
    state.focus = focus;
    preload();
    render();

    sceneGroup.querySelectorAll('[data-scene]').forEach((el) =>
      el.classList.toggle('is-active', el.dataset.scene === scene.id));

    focusGroup.innerHTML = '';
    scene.focus.forEach((f, i) => {
      const b = document.createElement('button');
      b.className = 'button' + (f === focus ? ' is-active' : '');
      b.textContent = 'Point ' + (i + 1);
      b.disabled = scene.focus.length === 1;
      b.onclick = () => selectView(scene, f);
      focusGroup.appendChild(b);
    });
  }

  slider.max = STOPS.length - 1;
  slider.value = state.stop;
  slider.addEventListener('input', () => {
    state.stop = parseInt(slider.value, 10);
    render();
  });

  SCENES.forEach((scene) => {
    const thumb = document.createElement('button');
    thumb.className = 'demo-thumb';
    thumb.dataset.scene = scene.id;
    thumb.innerHTML = `<img src="${src(scene, scene.focus[0], 'input')}" alt="Scene ${scene.id}" loading="lazy">`;
    if (REVIEW) {
      thumb.insertAdjacentHTML('beforeend', `<span class="demo-thumb-tag">${scene.id}</span>`);
    }
    thumb.onclick = () => selectView(scene, scene.focus[0]);
    sceneGroup.appendChild(thumb);
  });

  selectView(state.scene, state.focus);
})();

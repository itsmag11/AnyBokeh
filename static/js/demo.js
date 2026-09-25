(function () {
  const ROOT = 'static/images/demo';

  // `source` is the f-number of the input photo.
  const SCENES = [
    { id: '63', source: 'f/13', focus: ['focus1', 'focus2'] },
    { id: '71', source: 'f/10', focus: ['focus1', 'focus3'] },
    { id: '119', source: 'f/22', focus: ['focus1', 'focus2'] },
    { id: '51', source: 'f/2.0', focus: ['focus2'] },
    { id: '127', source: 'f/5.0', focus: ['focus2'] },
  ];

  const state = { scene: SCENES[0], focus: SCENES[0].focus[0], aperture: 'f2' };

  const inputImg = document.getElementById('demo-input');
  const inputLabel = document.getElementById('demo-input-label');
  const img = document.getElementById('demo-image');
  const label = document.getElementById('demo-label');
  const apertureGroup = document.getElementById('demo-apertures');
  const focusGroup = document.getElementById('demo-focus');
  const sceneGroup = document.getElementById('demo-scenes');

  const src = (scene, focus, aperture) => `${ROOT}/${scene.id}/${focus}_${aperture}.jpg`;

  function preload(scene) {
    scene.focus.forEach((f) =>
      ['input', 'f11', 'f4', 'f2'].forEach((a) => { new Image().src = src(scene, f, a); }));
  }

  function setActive(group, attr, value) {
    group.querySelectorAll('[data-' + attr + ']').forEach((el) =>
      el.classList.toggle('is-active', el.dataset[attr] === value));
  }

  function swap(el, next) {
    if (el.src.endsWith(next)) return;
    el.classList.add('is-fading');
    const loader = new Image();
    loader.onload = () => {
      el.src = next;
      el.classList.remove('is-fading');
    };
    loader.src = next;
  }

  function render() {
    swap(inputImg, src(state.scene, state.focus, 'input'));
    swap(img, src(state.scene, state.focus, state.aperture));
    inputLabel.textContent = `Input (${state.scene.source})`;
    label.textContent = 'AnyBokeh · f/' + state.aperture.slice(1);

    setActive(apertureGroup, 'aperture', state.aperture);
    setActive(sceneGroup, 'scene', state.scene.id);

    focusGroup.innerHTML = '';
    state.scene.focus.forEach((f, i) => {
      const b = document.createElement('button');
      b.className = 'button' + (f === state.focus ? ' is-active' : '');
      b.textContent = 'Point ' + (i + 1);
      b.disabled = state.scene.focus.length === 1;
      b.onclick = () => { state.focus = f; render(); };
      focusGroup.appendChild(b);
    });
  }

  apertureGroup.querySelectorAll('[data-aperture]').forEach((b) => {
    b.onclick = () => { state.aperture = b.dataset.aperture; render(); };
  });

  SCENES.forEach((scene) => {
    const thumb = document.createElement('button');
    thumb.className = 'demo-thumb';
    thumb.dataset.scene = scene.id;
    thumb.innerHTML = `<img src="${src(scene, scene.focus[0], 'input')}" alt="Scene ${scene.id}" loading="lazy">`;
    thumb.onclick = () => {
      state.scene = scene;
      state.focus = scene.focus[0];
      preload(scene);
      render();
    };
    sceneGroup.appendChild(thumb);
  });

  preload(state.scene);
  render();
})();

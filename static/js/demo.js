(function () {
  const ROOT = 'static/images/demo';

  // `source` is the f-number of the input photo; `id` is the image folder name.
  const SCENES = [
    { id: '63_f13', source: 'f/13', focus: ['focus1', 'focus2'] },
    { id: '71_f10', source: 'f/10', focus: ['focus1', 'focus2', 'focus3'] },
    { id: '119_f22', source: 'f/22', focus: ['focus1', 'focus2', 'focus3'] },
    { id: '119_f4.5', source: 'f/4.5', focus: ['focus1'] },
    { id: '51_f2.0', source: 'f/2.0', focus: ['focus1', 'focus2'] },
    { id: '127_f5.0', source: 'f/5.0', focus: ['focus2'] },
    { id: '36_f20', source: 'f/20', focus: ['focus1', 'focus2'] },
    { id: '72_f22', source: 'f/22', focus: ['focus1', 'focus2'] },
    { id: '86_f22', source: 'f/22', focus: ['focus2'] },
    { id: '109_f14', source: 'f/14', focus: ['focus2', 'focus3'] },
    { id: '109_f16', source: 'f/16', focus: ['focus3'] },
    { id: '157_f9.0', source: 'f/9.0', focus: ['focus1'] },
    { id: '165_f16', source: 'f/16', focus: ['focus1'] },
    { id: '168_f6.3', source: 'f/6.3', focus: ['focus3'] },
    { id: '63_f13_may', source: 'f/13', focus: ['focus1', 'focus2'] },
    { id: '71_f10_may', source: 'f/10', focus: ['focus1', 'focus3'] },
    { id: '119_f22_may', source: 'f/22', focus: ['focus1', 'focus2'] },
  ];

  const REVIEW = new URLSearchParams(location.search).has('review');

  const state = { scene: SCENES[0], focus: SCENES[0].focus[0], aperture: 'f2' };

  const inputImg = document.getElementById('demo-input');
  const inputLabel = document.getElementById('demo-input-label');
  const img = document.getElementById('demo-image');
  const label = document.getElementById('demo-label');
  const apertureGroup = document.getElementById('demo-apertures');
  const focusGroup = document.getElementById('demo-focus');
  const sceneGroup = document.getElementById('demo-scenes');

  const src = (scene, focus, aperture) => `${ROOT}/${scene.id}/${focus}_${aperture}.jpg?v=3`;

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
    if (REVIEW) {
      thumb.insertAdjacentHTML('beforeend', `<span class="demo-thumb-tag">${scene.id}</span>`);
    }
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

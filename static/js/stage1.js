(function () {
  const ROOT = 'static/images/stage1';
  const VERSION = 'v=1';

  const DATASETS = [
    {
      name: 'EBB!',
      scenes: [
        { id: 'ebb_4440', variants: [['bokeh', 'Bokeh'], ['original', 'All-in-focus']] },
        { id: 'ebb_4463', variants: [['bokeh', 'Bokeh'], ['original', 'All-in-focus']] },
        { id: 'ebb_4690', variants: [['bokeh', 'Bokeh'], ['original', 'All-in-focus']] },
      ],
    },
    {
      name: 'RealBokeh',
      scenes: [
        { id: 'rb_8', variants: [['f4.0', 'f/4.0'], ['f22', 'f/22']] },
        { id: 'rb_28', variants: [['f7.1', 'f/7.1'], ['f22', 'f/22']] },
        { id: 'rb_166', variants: [['f2.0', 'f/2.0'], ['f7.1', 'f/7.1']] },
      ],
    },
  ];
  const MAP_LABELS = { coc: 'Pred CoC', disp: 'Pred Disparity' };

  const state = { scene: DATASETS[0].scenes[0], map: 'coc' };

  const root = document.getElementById('stage1');
  const viewers = root.querySelectorAll('.compare');
  const ranges = root.querySelectorAll('.compare-range');
  const mapGroup = document.getElementById('stage1-maps');
  const sceneGroup = document.getElementById('stage1-scenes');

  const src = (scene, variant, kind) => `${ROOT}/${scene.id}/${variant}_${kind}.jpg?${VERSION}`;

  function preload(scene) {
    scene.variants.forEach(([v]) =>
      ['input', 'coc', 'disp'].forEach((k) => { new Image().src = src(scene, v, k); }));
  }

  // Both sliders share one position so the two inputs stay in sync.
  function setPos(value) {
    root.style.setProperty('--pos', value + '%');
    ranges.forEach((r) => { r.value = value; });
  }

  function render() {
    viewers.forEach((viewer, i) => {
      const [variant, label] = state.scene.variants[i];
      viewer.querySelector('.compare-top').src = src(state.scene, variant, 'input');
      viewer.querySelector('.compare-base').src = src(state.scene, variant, state.map);
      viewer.querySelector('.compare-label-left').textContent = `Input (${label})`;
      viewer.querySelector('.compare-label-right').textContent = MAP_LABELS[state.map];
    });

    mapGroup.querySelectorAll('[data-map]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.map === state.map));
    sceneGroup.querySelectorAll('[data-scene]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.scene === state.scene.id));
  }

  ranges.forEach((r) => r.addEventListener('input', () => setPos(r.value)));

  mapGroup.querySelectorAll('[data-map]').forEach((b) => {
    b.onclick = () => { state.map = b.dataset.map; render(); };
  });

  DATASETS.forEach((dataset) => {
    const group = document.createElement('div');
    group.className = 'stage1-group';
    group.innerHTML = `<span class="stage1-group-name">${dataset.name}</span>`;
    const row = document.createElement('div');
    row.className = 'stage1-thumbs';
    dataset.scenes.forEach((scene) => {
      const thumb = document.createElement('button');
      thumb.className = 'demo-thumb';
      thumb.dataset.scene = scene.id;
      thumb.innerHTML = `<img src="${src(scene, scene.variants[0][0], 'input')}" alt="${dataset.name} scene" loading="lazy">`;
      thumb.onclick = () => {
        state.scene = scene;
        preload(scene);
        render();
      };
      row.appendChild(thumb);
    });
    group.appendChild(row);
    sceneGroup.appendChild(group);
  });

  setPos(50);
  preload(state.scene);
  render();
})();

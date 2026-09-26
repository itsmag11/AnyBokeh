(function () {
  const ROOT = 'static/images/compare';
  const VERSION = 'v=2';

  const LABELS = {
    input: 'Input',
    ours: 'AnyBokeh (Ours)',
    restormer_bokehme: 'Restormer + BokehMe',
    restormer_bokehdiff: 'Restormer + BokehDiff',
    drbnet_bokehme: 'DRBNet + BokehMe',
    drbnet_bokehdiff: 'DRBNet + BokehDiff',
    bokehme: 'BokehMe',
    bokehdiff: 'BokehDiff',
    drbnet: 'DRBNet',
    restormer: 'Restormer',
    diffcamera: 'DiffCamera',
  };

  const A2A = ['restormer_bokehme', 'restormer_bokehdiff', 'drbnet_bokehme', 'drbnet_bokehdiff'];
  const BR = ['bokehme', 'bokehdiff'];
  const DD = ['drbnet', 'restormer', 'diffcamera'];

  // crop: [x, y, w, h] as fractions of the image, chosen where AnyBokeh differs most from the baselines.
  const TASKS = [
    {
      id: 'any2any',
      name: 'Any-to-Any Editing',
      cases: [
        { id: '201_f11_201_f2.0', setting: 'f/11 → f/2.0', baselines: A2A, crop: [0.685, 0.64, 0.3, 0.3] },
        { id: '66_f16_66_f2.5', setting: 'f/16 → f/2.5', baselines: A2A, crop: [0.3075, 0.26, 0.3, 0.3] },
        { id: '205_f4.0_205_f2.0', setting: 'f/4.0 → f/2.0', baselines: A2A, crop: [0.1713, 0.3433, 0.3, 0.3] },
        { id: '127_f5.0_127_f2.8', setting: 'f/5.0 → f/2.8', baselines: A2A, crop: [0.67, 0.3683, 0.3, 0.3] },
        { id: '186_f8.0_186_f14', setting: 'f/8.0 → f/14', baselines: A2A, crop: [0.685, 0.515, 0.3, 0.3] },
        { id: '25_f22_25_f2.0', setting: 'f/22 → f/2.0', baselines: A2A, crop: [0.6562, 0.5483, 0.3, 0.3] },
        { id: '119_f13_119_f4.0', setting: 'f/13 → f/4.0', baselines: A2A, crop: [0.1288, 0.49, 0.3, 0.3] },
        { id: '51_f2.0_51_f8.0', setting: 'f/2.0 → f/8.0', baselines: A2A, crop: [0.61, 0.6767, 0.3, 0.3] },
        { id: '71_f2.8_71_f14', setting: 'f/2.8 → f/14', baselines: A2A, crop: [0.5813, 0.675, 0.3, 0.3] },
        { id: '93_f22_93_f4.5', setting: 'f/22 → f/4.5', baselines: A2A, crop: [0.285, 0.09, 0.3, 0.3] },
      ],
    },
    {
      id: 'render',
      name: 'Bokeh Rendering',
      cases: [
        { id: '16_f2.0', setting: 'All-in-focus → f/2.0', baselines: BR, crop: [0.685, 0.2633, 0.3, 0.3] },
        { id: '43_f2.0', setting: 'All-in-focus → f/2.0', baselines: BR, crop: [0.015, 0.68, 0.3, 0.3] },
        { id: '4435_f1.8', setting: 'All-in-focus → f/1.8', baselines: BR, crop: [0.3475, 0.6317, 0.3, 0.3] },
        { id: '4401_f1.8', setting: 'All-in-focus → f/1.8', baselines: BR, crop: [0.6512, 0.44, 0.3, 0.3] },
        { id: '25_f2.2', setting: 'All-in-focus → f/2.2', baselines: BR, crop: [0.685, 0.52, 0.3, 0.3] },
        { id: '24_f5.0', setting: 'All-in-focus → f/5.0', baselines: BR, crop: [0.31, 0.02, 0.3, 0.3] },
        { id: '4578_f1.8', setting: 'All-in-focus → f/1.8', baselines: BR, crop: [0.1338, 0.03, 0.3, 0.3] },
        { id: '2_f2.0', setting: 'All-in-focus → f/2.0', baselines: BR, crop: [0.1237, 0.35, 0.3, 0.3] },
      ],
    },
    {
      id: 'deblur',
      name: 'Defocus Deblurring',
      cases: [
        { id: '4602_bokeh_aif', setting: 'Bokeh → All-in-focus', baselines: ['drbnet', 'restormer'], crop: [0.5437, 0.02, 0.3, 0.3] },
        { id: '13_f5.6_aif', setting: 'f/5.6 → All-in-focus', baselines: DD, crop: [0.015, 0.32, 0.3, 0.3] },
        { id: '4416_bokeh_aif', setting: 'Bokeh → All-in-focus', baselines: ['drbnet', 'restormer'], crop: [0.1338, 0.2033, 0.3, 0.3] },
        { id: '12_f2.0_aif', setting: 'f/2.0 → All-in-focus', baselines: DD, crop: [0.2425, 0.6583, 0.3, 0.3] },
        { id: '6_f10_aif', setting: 'f/10 → All-in-focus', baselines: DD, crop: [0.11, 0.0217, 0.3, 0.3] },
        { id: '8_f4.0_aif', setting: 'f/4.0 → All-in-focus', baselines: DD, crop: [0.1812, 0.25, 0.3, 0.3] },
        { id: '28_f7.1_aif', setting: 'f/7.1 → All-in-focus', baselines: DD, crop: [0.2025, 0.2517, 0.3, 0.3] },
      ],
    },
  ];

  const root = document.getElementById('cmp');
  const viewer = root.querySelector('.cmp-viewer');
  const range = root.querySelector('.compare-range');
  const taskGroup = document.getElementById('cmp-tasks');
  const sceneGroup = document.getElementById('cmp-scenes');
  const setting = document.getElementById('cmp-setting');
  const box = document.getElementById('cmp-box');
  const crops = document.getElementById('cmp-crops');

  const state = { task: TASKS[0], scene: TASKS[0].cases[0], method: TASKS[0].cases[0].baselines[0] };

  const src = (scene, name) => `${ROOT}/${state.task.id}/${scene.id}/${name}.jpg?${VERSION}`;
  const leftOptions = (scene) => ['input', ...scene.baselines];

  function preload(scene) {
    ['ours', 'gt', ...leftOptions(scene)].forEach((m) => {
      new Image().src = src(scene, m);
      new Image().src = src(scene, `crop_${m}`);
    });
  }

  function setPos(value) {
    root.style.setProperty('--pos', value + '%');
    range.value = value;
  }

  function buildCrops() {
    crops.innerHTML = '';
    [...leftOptions(state.scene), 'ours', 'gt'].forEach((m) => {
      const fixed = m === 'ours' || m === 'gt';
      const tile = document.createElement(fixed ? 'div' : 'button');
      tile.className = 'cmp-crop' + (fixed ? ` is-${m}` : '');
      if (!fixed) tile.dataset.method = m;
      const label = m === 'gt' ? 'GT' : LABELS[m];
      tile.innerHTML = `<img src="${src(state.scene, `crop_${m}`)}" alt="${label} crop"><span>${label}</span>`;
      if (!fixed) tile.onclick = () => { state.method = m; render(); };
      crops.appendChild(tile);
    });
  }

  function render() {
    const { scene, method } = state;
    viewer.querySelector('.compare-base').src = src(scene, 'ours');
    viewer.querySelector('.compare-top').src = src(scene, method);
    viewer.querySelector('.compare-label-left').textContent = LABELS[method];
    setting.textContent = scene.setting;

    const [x, y, w, h] = scene.crop;
    Object.assign(box.style, { left: `${x * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${h * 100}%` });

    taskGroup.querySelectorAll('[data-task]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.task === state.task.id));
    crops.querySelectorAll('[data-method]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.method === method));
    sceneGroup.querySelectorAll('[data-scene]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.scene === scene.id));
  }

  function selectScene(scene) {
    state.scene = scene;
    if (!leftOptions(scene).includes(state.method)) state.method = scene.baselines[0];
    buildCrops();
    preload(scene);
    render();
  }

  function buildTask() {
    sceneGroup.innerHTML = '';
    state.task.cases.forEach((scene) => {
      const thumb = document.createElement('button');
      thumb.className = 'demo-thumb';
      thumb.dataset.scene = scene.id;
      thumb.innerHTML = `<img src="${src(scene, 'ours')}" alt="${state.task.name} scene" loading="lazy">`;
      thumb.onclick = () => selectScene(scene);
      sceneGroup.appendChild(thumb);
    });
  }

  TASKS.forEach((task) => {
    const b = document.createElement('button');
    b.className = 'cmp-task';
    b.dataset.task = task.id;
    b.textContent = task.name;
    b.onclick = () => {
      if (state.task === task) return;
      state.task = task;
      state.method = task.cases[0].baselines[0];
      buildTask();
      selectScene(task.cases[0]);
    };
    taskGroup.appendChild(b);
  });

  range.addEventListener('input', () => setPos(range.value));

  setPos(50);
  buildTask();
  selectScene(state.scene);
})();

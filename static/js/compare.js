(function () {
  const ROOT = 'static/images/compare';
  const VERSION = 'v=1';

  const TASKS = [
    {
      id: 'any2any',
      name: 'Any-to-Any Editing',
      methods: [
        ['input', 'Input'],
        ['restormer_bokehme', 'Restormer + BokehMe'],
        ['restormer_bokehdiff', 'Restormer + BokehDiff'],
        ['drbnet_bokehme', 'DRBNet + BokehMe'],
        ['drbnet_bokehdiff', 'DRBNet + BokehDiff'],
        ['gt', 'GT'],
      ],
      cases: [
        { id: '109_f2.8_109_f6.3', setting: 'f/2.8 → f/6.3' },
        { id: '93_f22_93_f4.5', setting: 'f/22 → f/4.5' },
        { id: '4435_bokeh_4435_original', setting: 'Bokeh → All-in-focus' },
        { id: '4459_original_4459_bokeh', setting: 'All-in-focus → Bokeh' },
        { id: '27_f18_27_f7.1', setting: 'f/18 → f/7.1' },
        { id: '36_f10_36_f6.3', setting: 'f/10 → f/6.3' },
        { id: '39_f7.1_39_f3.2', setting: 'f/7.1 → f/3.2' },
        { id: '76_f2.5_76_f5.6', setting: 'f/2.5 → f/5.6' },
      ],
    },
    {
      id: 'render',
      name: 'Bokeh Rendering',
      methods: [
        ['input', 'Input'],
        ['bokehme', 'BokehMe'],
        ['bokehdiff', 'BokehDiff'],
        ['gt', 'GT'],
      ],
      cases: [
        { id: '10_f2.0', setting: 'All-in-focus → f/2.0' },
        { id: '37_f2.0', setting: 'All-in-focus → f/2.0' },
        { id: '39_f4.0', setting: 'All-in-focus → f/4.0' },
      ],
    },
    {
      id: 'deblur',
      name: 'Defocus Deblurring',
      methods: [
        ['input', 'Input'],
        ['drbnet', 'DRBNet'],
        ['restormer', 'Restormer'],
        ['gt', 'GT'],
      ],
      cases: [
        { id: '13_f5.6_aif', setting: 'f/5.6 → All-in-focus' },
        { id: '1_f11_aif', setting: 'f/11 → All-in-focus' },
        { id: '9_f2.0_aif', setting: 'f/2.0 → All-in-focus' },
      ],
    },
  ];

  const root = document.getElementById('cmp');
  const viewer = root.querySelector('.cmp-viewer');
  const range = root.querySelector('.compare-range');
  const taskGroup = document.getElementById('cmp-tasks');
  const methodGroup = document.getElementById('cmp-methods');
  const sceneGroup = document.getElementById('cmp-scenes');
  const setting = document.getElementById('cmp-setting');

  const state = { task: TASKS[0], scene: TASKS[0].cases[0], method: TASKS[0].methods[1][0] };

  const src = (task, scene, method) => `${ROOT}/${task.id}/${scene.id}/${method}.jpg?${VERSION}`;

  function preload(task, scene) {
    ['ours', ...task.methods.map(([m]) => m)].forEach((m) => { new Image().src = src(task, scene, m); });
  }

  function setPos(value) {
    root.style.setProperty('--pos', value + '%');
    range.value = value;
  }

  function render() {
    const { task, scene, method } = state;
    viewer.querySelector('.compare-base').src = src(task, scene, 'ours');
    viewer.querySelector('.compare-top').src = src(task, scene, method);
    viewer.querySelector('.compare-label-left').textContent =
      task.methods.find(([m]) => m === method)[1];
    setting.textContent = scene.setting;

    taskGroup.querySelectorAll('[data-task]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.task === task.id));
    methodGroup.querySelectorAll('[data-method]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.method === method));
    sceneGroup.querySelectorAll('[data-scene]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.scene === scene.id));
  }

  function buildTask() {
    const { task } = state;
    methodGroup.innerHTML = '';
    task.methods.forEach(([m, label]) => {
      const b = document.createElement('button');
      b.className = 'cmp-method';
      b.dataset.method = m;
      b.textContent = label;
      b.onclick = () => { state.method = m; render(); };
      methodGroup.appendChild(b);
    });

    sceneGroup.innerHTML = '';
    task.cases.forEach((scene) => {
      const thumb = document.createElement('button');
      thumb.className = 'demo-thumb';
      thumb.dataset.scene = scene.id;
      thumb.innerHTML = `<img src="${src(task, scene, 'ours')}" alt="${task.name} scene" loading="lazy">`;
      thumb.onclick = () => { state.scene = scene; preload(task, scene); render(); };
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
      state.scene = task.cases[0];
      state.method = task.methods[1][0];
      buildTask();
      preload(task, state.scene);
      render();
    };
    taskGroup.appendChild(b);
  });

  range.addEventListener('input', () => setPos(range.value));

  setPos(50);
  buildTask();
  preload(state.task, state.scene);
  render();
})();

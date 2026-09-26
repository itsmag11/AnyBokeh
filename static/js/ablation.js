(function () {
  const ROOT = 'static/images/ablation';
  const VERSION = 'v=1';

  const SCENES = [
    '14_f2.0_14_f10',
    '27_f2.8_27_f20',
    '39_f2.5_39_f7.1',
    '33_f3.5_33_f2.0',
    '96_f3.2_96_f4.0',
    '176_f2.0_176_f2.5',
  ];

  const ABLATIONS = [
    {
      id: 'single',
      name: 'One-Stage vs. Two-Stage',
      label: 'Single-stage',
      desc: 'Single-stage computes both CoC maps from Depth Pro depth and EXIF, with bokeh-level calibration. '
        + 'AnyBokeh predicts source CoC and disparity jointly in Stage 1, learning blur-aware geometry.',
      srcCoc: true,
    },
    {
      id: 'nosrc',
      name: 'Without vs. With Source CoC',
      label: 'W/o source CoC',
      desc: 'Same target CoC, but the editor no longer gets the source CoC and must guess the existing blur from appearance alone.',
      srcCoc: false,
    },
  ];

  const root = document.getElementById('abl');
  const viewer = root.querySelector('.abl-viewer');
  const range = viewer.querySelector('.compare-range');
  const taskGroup = document.getElementById('abl-tasks');
  const desc = document.getElementById('abl-desc');
  const switchGroup = document.getElementById('abl-switch');
  const grid = document.getElementById('abl-grid');
  const sceneGroup = document.getElementById('abl-scenes');
  const setting = document.getElementById('abl-setting');

  const state = { ablation: ABLATIONS[0], scene: SCENES[0], left: 'variant' };

  const src = (scene, name) => `${ROOT}/${scene}/${name}.jpg?${VERSION}`;

  function settingText(scene) {
    const [, a, , b] = scene.split('_');
    return `${a.replace('f', 'f/')} → ${b.replace('f', 'f/')}`;
  }

  function leftOptions() {
    return [
      ['variant', state.ablation.label, state.ablation.id],
      ['input', 'Input', 'input'],
      ['gt', 'GT', 'gt'],
    ];
  }

  function setPos(value) {
    root.style.setProperty('--pos', value + '%');
    range.value = value;
  }

  function cell(img, caption) {
    return img
      ? `<figure class="abl-cell"><img src="${img}" alt="${caption}"><figcaption>${caption}</figcaption></figure>`
      : `<figure class="abl-cell is-empty"><div class="abl-empty">Not used</div><figcaption>${caption}</figcaption></figure>`;
  }

  function renderGrid() {
    const { ablation, scene } = state;
    const v = ablation.id;
    grid.innerHTML = `
      <span class="abl-row-label">${ablation.label}</span>
      ${cell(ablation.srcCoc ? src(scene, `${v}_src_coc`) : null, 'Source CoC')}
      ${cell(src(scene, `${v}_tgt_coc`), 'Target CoC')}
      <span class="abl-row-label is-ours">AnyBokeh</span>
      ${cell(src(scene, 'ours_src_coc'), 'Source CoC')}
      ${cell(src(scene, 'ours_tgt_coc'), 'Target CoC')}`;
  }

  function render() {
    const { ablation, scene, left } = state;
    const option = leftOptions().find(([k]) => k === left);
    viewer.querySelector('.compare-base').src = src(scene, 'ours');
    viewer.querySelector('.compare-top').src = src(scene, option[2]);
    viewer.querySelector('.compare-label-left').textContent = option[1];
    setting.textContent = settingText(scene);
    desc.textContent = ablation.desc;

    switchGroup.innerHTML = '';
    leftOptions().forEach(([k, label]) => {
      const b = document.createElement('button');
      b.className = 'abl-switch-btn' + (k === left ? ' is-active' : '');
      b.textContent = label;
      b.onclick = () => { state.left = k; render(); };
      switchGroup.appendChild(b);
    });

    renderGrid();

    taskGroup.querySelectorAll('[data-task]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.task === ablation.id));
    sceneGroup.querySelectorAll('[data-scene]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.scene === scene));
  }

  ABLATIONS.forEach((ablation) => {
    const b = document.createElement('button');
    b.className = 'cmp-task';
    b.dataset.task = ablation.id;
    b.textContent = ablation.name;
    b.onclick = () => { state.ablation = ablation; state.left = 'variant'; render(); };
    taskGroup.appendChild(b);
  });

  SCENES.forEach((scene) => {
    const thumb = document.createElement('button');
    thumb.className = 'demo-thumb';
    thumb.dataset.scene = scene;
    thumb.innerHTML = `<img src="${src(scene, 'input')}" alt="Ablation scene" loading="lazy">`;
    thumb.onclick = () => { state.scene = scene; render(); };
    sceneGroup.appendChild(thumb);
  });

  range.addEventListener('input', () => setPos(range.value));

  setPos(50);
  render();
})();

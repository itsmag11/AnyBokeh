(function () {
  const DURATION = 7000;

  const root = document.getElementById('method');
  const tabs = root.querySelectorAll('.method-tab');
  const masks = root.querySelectorAll('.method-mask');
  const captions = root.querySelectorAll('.method-caption');

  let step = 0;
  let timer = null;
  let paused = false;
  let visible = false;

  function show(i) {
    step = i;
    tabs.forEach((t, k) => {
      t.classList.toggle('is-active', k === i);
      const bar = t.querySelector('.method-progress');
      // Restart the progress-bar animation.
      bar.style.animation = 'none';
      void bar.offsetWidth;
      bar.style.animation = '';
    });
    masks.forEach((m, k) => m.classList.toggle('is-dimmed', k !== i));
    captions.forEach((c, k) => c.classList.toggle('is-active', k === i));
    schedule();
  }

  function schedule() {
    clearTimeout(timer);
    root.classList.toggle('is-paused', paused || !visible);
    if (!paused && visible) timer = setTimeout(() => show((step + 1) % tabs.length), DURATION);
  }

  root.style.setProperty('--method-duration', DURATION + 'ms');
  tabs.forEach((t, k) => { t.onclick = () => show(k); });
  root.addEventListener('mouseenter', () => { paused = true; schedule(); });
  root.addEventListener('mouseleave', () => { paused = false; show(step); });

  new IntersectionObserver(([entry]) => {
    const wasVisible = visible;
    visible = entry.isIntersecting;
    if (visible && !wasVisible) show(0);
    else schedule();
  }, { threshold: 0.4 }).observe(root);

  show(0);
})();

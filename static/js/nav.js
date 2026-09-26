(function () {
  const links = document.querySelectorAll('.topnav-links a[href^="#"]');
  const sections = [...links].map((a) => document.querySelector(a.getAttribute('href')));

  function update() {
    const y = window.scrollY + window.innerHeight * 0.35;
    let current = -1;
    sections.forEach((s, i) => { if (s.getBoundingClientRect().top + window.scrollY <= y) current = i; });
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) current = sections.length - 1;
    links.forEach((a, i) => a.classList.toggle('is-active', i === current));
  }

  window.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
  update();
})();

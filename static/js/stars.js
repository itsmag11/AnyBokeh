(function () {
  const REPO = 'itsmag11/AnyBokeh';
  const CACHE_KEY = 'anybokeh-stars';
  const CACHE_MS = 60 * 60 * 1000;

  const format = (n) => (n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1).replace(/\.0$/, '') + 'k' : String(n));

  function show(count) {
    document.querySelectorAll('[data-gh-stars]').forEach((el) => {
      el.querySelector('b').textContent = format(count);
      el.hidden = false;
    });
  }

  // The unauthenticated GitHub API allows 60 requests per hour per visitor IP, so cache locally.
  try {
    const cached = JSON.parse(localStorage.getItem(CACHE_KEY) || 'null');
    if (cached && Date.now() - cached.time < CACHE_MS) {
      show(cached.count);
      return;
    }
  } catch (e) { /* ignore malformed cache */ }

  fetch(`https://api.github.com/repos/${REPO}`)
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((data) => {
      const count = data.stargazers_count;
      if (typeof count !== 'number') return;
      show(count);
      try { localStorage.setItem(CACHE_KEY, JSON.stringify({ count, time: Date.now() })); } catch (e) { /* storage unavailable */ }
    })
    .catch(() => {});
})();

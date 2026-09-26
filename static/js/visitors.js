(function () {
  const STATS_URL = 'https://mapmyvisitors.com/web/1c8gm';

  // The widget injects its link asynchronously, so poll briefly until it appears.
  let tries = 0;
  const timer = setInterval(() => {
    const link = document.getElementById('mapmyvisitors-widget');
    if (link) {
      link.href = STATS_URL;
      link.target = '_blank';
      link.rel = 'noopener';
    }
    if (++tries > 40) clearInterval(timer);
  }, 250);
})();

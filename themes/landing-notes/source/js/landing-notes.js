(() => {
  const button = document.querySelector('.menu-toggle');
  const nav = document.querySelector('#main-nav');
  if (!button || !nav) return;
  const setOpen = open => {
    button.setAttribute('aria-expanded', String(open));
    button.textContent = open ? '关闭 −' : '菜单 +';
    nav.classList.toggle('is-open', open);
  };
  button.addEventListener('click', () => setOpen(button.getAttribute('aria-expanded') !== 'true'));
  nav.addEventListener('click', event => { if (event.target.closest('a')) setOpen(false); });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && button.getAttribute('aria-expanded') === 'true') { setOpen(false); button.focus(); }
  });
})();

// Each filter owns its list; all published posts are present across pagination.
document.querySelectorAll('[data-tag-browser]').forEach(browser => {
  const buttons = [...browser.querySelectorAll('button[data-tag]')];
  const rows = [...browser.querySelectorAll('[data-filter-row]')].map(element => ({element, tags: JSON.parse(element.dataset.tags)}));
  const pageSize = Math.max(1, Number(browser.dataset.pageSize) || 10);
  let selected = browser.dataset.initialTag || '';
  let limit = pageSize;
  const more = browser.querySelector('[data-filter-more]');
  const render = () => {
    const matches = rows.filter(row => !selected || row.tags.includes(selected));
    rows.forEach(row => { row.element.hidden = true; });
    matches.slice(0, limit).forEach((row, i) => {
      row.element.hidden = false;
      row.element.querySelector('[data-row-number]').textContent = String(i + 1).padStart(2, '0');
    });
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.tag === selected)));
    browser.querySelector('[data-filter-status]').textContent = `${selected || '全部'} · ${matches.length} 篇文章`;
    browser.querySelector('[data-filter-empty]').hidden = matches.length !== 0;
    more.hidden = matches.length <= limit;
    more.textContent = `显示更多 · 还有 ${Math.max(0, matches.length - limit)} 篇`;
    const featured = browser.parentElement.querySelector('.featured-article');
    if (featured) featured.hidden = Boolean(selected);
  };
  buttons.forEach(button => button.addEventListener('click', () => {
    selected = button.dataset.tag;
    limit = pageSize;
    render();
  }));
  more.addEventListener('click', () => { limit += pageSize; render(); });
  render();
  browser.hidden = false;
  browser.parentElement.querySelectorAll('[data-tag-fallback]').forEach(element => { element.hidden = true; });
});

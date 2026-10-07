'use strict';

// Protect ordinary $...$ before Markdown interprets TeX underscores/escapes.
// Keep Kramed's existing $$...$$ and legacy `$...$` handling.
const kramed = require('kramed');
const rules = kramed.InlineLexer.rules;
const singleDollar = /^\$(?!\$|\s)((?:\\[^\n]|[^\\$\n])+?)\$(?!\$)/;
for (const name of ['normal', 'pedantic', 'gfm', 'breaks', 'gitbook']) {
  const original = rules[name].math;
  rules[name].math = {
    exec(text) {
      return original.exec(text) || singleDollar.exec(text);
    }
  };
}

// hexo-filter-mathjax scans text delimiters, not Kramed's math/tex scripts.
// Escape HTML only; leave TeX intact for server-side SVG typesetting.
kramed.Renderer.prototype.math = function(content, language, display) {
  const escaped = content.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return display ? '<div>\\[' + escaped + '\\]</div>' : '<span>\\(' + escaped + '\\)</span>';
};

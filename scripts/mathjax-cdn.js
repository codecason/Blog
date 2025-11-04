// scripts/mathjax-cdn.js
hexo.extend.filter.register('after_render:html', function(htmlContent) {
    const oldCdn = /http?:\/\/.*?mathjax\/.*?\.js/g;
    const newCdn = 'https://cdn.bootcdn.net/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js';
    return htmlContent.replace(oldCdn, newCdn);
  });

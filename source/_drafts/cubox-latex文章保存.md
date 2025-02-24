一些公式下载下来之后是
L_{DPO}(\\pi_{\\theta}; \\pi_{ref}) = -\\mathbb{E}_{(x, y_w, y_l)\\sim D}\[log\\sigma\[\\beta log\\frac{\\pi_{\\theta}(y_w\|x)}{\\pi_{ref}(y_w\|x)} - \\beta log\\frac{\\pi_{\\theta}(y_l\|x)}{\\pi_{ref}(y_l\|x)}\]\]   

需要:
1.删除重复公式;
2.把两个斜杠\\转成\；
3.增加公式符号 $$包裹;

博客需要安装渲染引擎kramed，同时把hexo-math替换成hexo-renderer-math.

其他:
把图片转存到本地并修改路径。

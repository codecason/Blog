#### 支持Mathjax https://myblackboxrecorder.com/use-math-in-hexo/

Step
npm uninstall hexo-math --save
npm uninstall hexo-renderer-mathjax --save

Step
npm install hexo-filter-mathjax --save

对部分支持MathJax的主题来说，只需在主题配置文件将相关配置项开启即可使用MathJax。但对很多主题，需要自己配置。hexo-filter-mathjax可以帮我们解决这个问题。


渲染引擎

npm install hexo-renderer-kramed --save

escape: 
/^\\([\\`*{}\[\]()#$+\-.!_>])/
escape:
/^\\([`*\[\]()# +\-.!_>])/,

意味着把`\#, \*`等符号改写成字面的符号；

(env310) /mnt/d/Labs/Labs-11/Blog$ npm list | grep math
├── hexo-filter-mathjax@0.9.0
├── hexo-math@5.0.0
(env310) /mnt/d/Labs/Labs-11/Blog$ npm list | grep renderer
├── hexo-renderer-ejs@2.0.0
├── hexo-renderer-marked@7.0.0
├── hexo-renderer-stylus@3.0.1

\$会被解释成 $直接写出来；
但是\\不会被解释成？

#### 油猴脚本爬取知乎专栏文章并转换数学公式为latex格式

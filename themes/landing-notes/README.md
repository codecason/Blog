# Landing Notes — Hexo theme

This theme adapts the typography, paper/ink/accent palette, article lists and reader styling of https://github.com/LandingW/landingw.github.io/tree/283d03c087cd21d5dd121de0fb324335205bb7da to Hexo. Original CSS provenance remains in source/css/landing-notes.css. Upstream does not declare a general repository license; this local adaptation does not grant additional rights to upstream material.

## 日常使用

文章继续放在 source/_posts，草稿继续放在 source/_drafts；原有日期链接、标签、归档、代码高亮和构建时数学公式保持 Hexo 工作流。
站点标题、作者、描述在根目录 _config.yml；主题首页标题、介绍、品牌文字在 themes/landing-notes/_config.yml。也可创建根目录 _config.landing-notes.yml 来覆盖主题设置。
导航使用现有首页、archives、tags、about 页面。所有主题资源和内部链接经 url_for 处理，兼容 /Blog/。

```powershell
npm run build
npm run server -- --port 4318 --ip 127.0.0.1
```

访问 http://127.0.0.1:4318/Blog/ 。远程发布仍由原 .github/workflows/pages.yml 处理。

## 备份与回滚

旧主题完整保存在 themes/even.backup-2026-10-08，原站点配置保存在 _config.yml.backup-2026-10-08。
最快切回旧外观：将根配置中的 theme 改为 even.backup-2026-10-08，然后 npm run clean 与 npm run build。
需要恢复旧名称时，在确认 themes/even 不存在后，把备份目录改回 even，并将 theme 改回 even。
只有确定不需要保留后续配置修改时，才用备份配置覆盖 _config.yml。

## 范围

这是 Hexo 主题，不是把 Next.js 嵌进 Hexo。首页使用轻量几何 SVG，未移植 React/Three.js 的茶壶、分形播放或文字噪声动画。

验证：node tests/theme.cjs、node tests/math.cjs、npm run build。

首页和标签页现在支持原地标签筛选：点击标签切换，点击全部恢复；筛选覆盖全部已发布文章，超过每页数量时可显示更多。禁用 JavaScript 时保留原分页和标签链接。

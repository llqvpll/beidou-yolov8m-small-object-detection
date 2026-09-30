const fs = require('fs');
const html = fs.readFileSync('index.html', 'utf8');
const app = fs.readFileSync('assets/js/app.js', 'utf8');
const vis = fs.readFileSync('assets/js/visual.js', 'utf8');
const css = fs.readFileSync('assets/css/app.css', 'utf8');

const ids = new Set();
let m; const reId = /\sid="([^"]+)"/g;
while ((m = reId.exec(html))) ids.add(m[1]);

const refs = new Set();
[app, vis].forEach(src => {
  let x;
  const r1 = /\$\('#([A-Za-z0-9_\-]+)'/g;
  while ((x = r1.exec(src))) refs.add(x[1]);
  const r2 = /getElementById\('([A-Za-z0-9_\-]+)'\)/g;
  while ((x = r2.exec(src))) refs.add(x[1]);
});
const missing = [...refs].filter(r => !ids.has(r));

const syms = new Set(); const reSym = /<g id="(i-[a-z0-9\-]+)"/g;
while ((m = reSym.exec(html))) syms.add(m[1]);
const used = new Set();
[html, app, vis].forEach(src => { let x; const r = /#(i-[a-z0-9\-]+)/g; while ((x = r.exec(src))) used.add(x[1]); });
const badIcons = [...used].filter(u => !syms.has(u));
const unusedIcons = [...syms].filter(u => !used.has(u));

const jsClasses = new Set();
[app, vis].forEach(src => {
  let x; const r = /class="([^"$]+)"/g;
  while ((x = r.exec(src))) x[1].split(/\s+/).filter(Boolean).forEach(c => jsClasses.add(c));
});
const missingCss = [...jsClasses].filter(c => !css.includes('.' + c));

console.log('HTML ids:', ids.size, '| JS refs:', refs.size, '| icons:', syms.size, '| used:', used.size);
console.log('MISSING IDS      :', missing.length ? missing.join(', ') : '(none)');
console.log('BAD ICONS        :', badIcons.length ? badIcons.join(', ') : '(none)');
console.log('UNUSED ICONS     :', unusedIcons.length ? unusedIcons.join(', ') : '(none)');
console.log('JS CLASSES NO CSS:', missingCss.length ? missingCss.join(', ') : '(none)');

// HTML 中 class 引用是否在 CSS 定义
const htmlClasses = new Set();
const rc = /class="([^"]+)"/g;
while ((m = rc.exec(html))) m[1].split(/\s+/).filter(Boolean).forEach(c => htmlClasses.add(c));
const htmlNoCss = [...htmlClasses].filter(c => !css.includes('.' + c) && !/^(is-|has-)/.test(c));
console.log('HTML CLASSES NO CSS:', htmlNoCss.length ? htmlNoCss.join(', ') : '(none)');

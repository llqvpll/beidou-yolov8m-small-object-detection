const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.error('NO SCRIPT FOUND'); process.exit(2); }
const code = m[1];
try {
  new vm.Script(code, { filename: 'inline.js' });
  console.log('JS SYNTAX OK (' + code.split('\n').length + ' lines)');
} catch (e) {
  console.error('JS SYNTAX ERROR:', e.message);
  process.exit(1);
}
// quick checks for required ids referenced by $()
const ids = [...code.matchAll(/\$\('#([\w-]+)'\)/g)].map(x => x[1]);
const uniqueIds = [...new Set(ids)];
const htmlHas = uniqueIds.filter(id => !new RegExp('id="' + id + '"').test(html));
console.log('Referenced ids:', uniqueIds.length, '| missing in HTML:', htmlHas.length ? htmlHas.join(',') : 'none');

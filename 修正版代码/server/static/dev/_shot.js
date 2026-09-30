/* 极简 CDP 截图 / 探针：node _shot.js <url> <out.png> [w] [h] [scale] [waitMs] [evalFile] */
const http = require('http');
const fs = require('fs');
const PORT = process.env.CDP_PORT || 9333;

function getJSON(path) {
  return new Promise((res, rej) => {
    http.get({ host: '127.0.0.1', port: PORT, path }, r => {
      let d = '';
      r.on('data', c => (d += c));
      r.on('end', () => { try { res(JSON.parse(d)); } catch (e) { rej(e); } });
    }).on('error', rej);
  });
}

class CDP {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.id = 0;
    this.pending = new Map();
    this.logs = [];
    this.ws.addEventListener('message', ev => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const p = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) p.rej(new Error(JSON.stringify(msg.error)));
        else p.res(msg.result);
        return;
      }
      if (msg.method === 'Runtime.consoleAPICalled') {
        this.logs.push('[' + msg.params.type + '] ' +
          (msg.params.args || []).map(a => a.value !== undefined ? a.value : (a.description || a.type)).join(' '));
      }
      if (msg.method === 'Runtime.exceptionThrown') {
        const d = msg.params.exceptionDetails || {};
        let s = '[EXCEPTION] ' + (d.exception && (d.exception.description || d.exception.value) || d.text);
        const frames = (d.stackTrace && d.stackTrace.callFrames) || [];
        s += '\n    at ' + frames.slice(0, 4).map(f => f.functionName + ' (' + (f.url || '').split('/').pop() + ':' + (f.lineNumber + 1) + ')').join('\n    at ');
        this.logs.push(s);
      }
      if (msg.method === 'Log.entryAdded') {
        const e = msg.params.entry;
        if (e.level === 'error' || e.level === 'warning') this.logs.push('[' + e.level + '] ' + e.text);
      }
    });
  }
  ready() { return new Promise((res, rej) => { this.ws.addEventListener('open', () => res()); this.ws.addEventListener('error', e => rej(e)); }); }
  send(method, params) {
    const id = ++this.id;
    return new Promise((res, rej) => {
      this.pending.set(id, { res, rej });
      this.ws.send(JSON.stringify({ id, method, params: params || {} }));
      setTimeout(() => { if (this.pending.has(id)) { this.pending.delete(id); rej(new Error('timeout: ' + method)); } }, +(process.env.CDP_TIMEOUT || 30000));
    });
  }
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const url = process.argv[2];
  const out = process.argv[3];
  const W = +(process.argv[4] || 1600);
  const H = +(process.argv[5] || 1000);
  const SCALE = +(process.argv[6] || 2);
  const WAIT = +(process.argv[7] || 1400);
  const evalFile = process.argv[8];

  const targets = await getJSON('/json/list');
  const page = targets.find(t => t.type === 'page');
  if (!page) { console.error('no page target'); process.exit(1); }

  const cdp = new CDP(page.webSocketDebuggerUrl);
  await cdp.ready();
  await cdp.send('Runtime.enable');
  await cdp.send('Log.enable');
  await cdp.send('Page.enable');
  try { await cdp.send('Network.enable'); await cdp.send('Network.setCacheDisabled', { cacheDisabled: true }); } catch (e) { }
  await cdp.send('Emulation.setDeviceMetricsOverride', {
    width: W, height: H, deviceScaleFactor: SCALE, mobile: false,
  });

  // 先回到 about:blank，确保每次都完整重载（避免仅 fragment 变化导致的同文档导航）
  await cdp.send('Page.navigate', { url: 'about:blank' });
  await sleep(220);
  await cdp.send('Page.navigate', { url });
  await sleep(WAIT);

  if (evalFile) {
    const expr = fs.readFileSync(evalFile, 'utf8');
    try {
      const r = await cdp.send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
      if (r && r.result) console.log('EVAL:', JSON.stringify(r.result.value));
    } catch (e) { console.log('EVAL ERR:', e.message); }
    await sleep(1200);
  }

  const shotParams = { format: 'png', captureBeyondViewport: false };
  if (process.env.CLIP) {
    const p = process.env.CLIP.split(',').map(Number);
    shotParams.clip = { x: p[0], y: p[1], width: p[2], height: p[3], scale: 1 };
  }
  const shot = await cdp.send('Page.captureScreenshot', shotParams);
  fs.writeFileSync(out, Buffer.from(shot.data, 'base64'));

  if (cdp.logs.length) {
    console.log('--- CONSOLE ---');
    cdp.logs.slice(0, 40).forEach(l => console.log(l));
  } else {
    console.log('--- CONSOLE: clean ---');
  }
  console.log('SAVED:', out);
  process.exit(0);
})().catch(e => { console.error('FATAL:', e.message); process.exit(1); });

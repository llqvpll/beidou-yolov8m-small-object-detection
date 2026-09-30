/* ============================================================================
 * app.js — 应用层
 *   视图路由 / 上传与队列 / 推理调度 / 舞台交互 / 结果面板
 *   任务 / 分析 / 北斗 / 设置 四个视图 / 命令面板 / 快捷键 / 主题
 * ========================================================================== */
(function (global) {
  'use strict';

  const BDS = global.BDS;
  const U = BDS.util, Store = BDS.Store, Api = BDS.Api, Overlay = BDS.Overlay,
    Charts = BDS.Charts, Beidou = BDS.Beidou, Demo = BDS.Demo, CLASSES = BDS.CLASSES;
  const S = Store.state;
  const $ = U.$, $$ = U.$$;
  const isMac = U.isMac();

  /* ======================================================== Toast */
  const TOAST_ICON = { ok: 'i-check', warn: 'i-alert', bad: 'i-alert', info: 'i-info' };
  function toast(type, title, desc, ms) {
    const host = $('#toasts');
    const el = document.createElement('div');
    el.className = 'toast ' + (type || 'info');
    el.innerHTML =
      '<svg><use href="#' + (TOAST_ICON[type] || 'i-info') + '"/></svg>' +
      '<div class="sp"><div class="tt">' + U.esc(title) + '</div>' +
      (desc ? '<div class="td">' + U.esc(desc) + '</div>' : '') + '</div>' +
      '<button class="ibtn sm"><svg><use href="#i-x"/></svg></button>';
    const close = () => {
      el.classList.add('out');
      setTimeout(() => el.remove(), 220);
    };
    el.querySelector('button').onclick = close;
    host.appendChild(el);
    setTimeout(close, ms || 4200);
  }
  BDS.toast = toast;
  BDS.go = go;                   // 暴露给 dev 探针与调试用

  /* ======================================================== 视图路由 */
  const VIEWS = ['detect', 'jobs', 'analytics', 'geo', 'settings'];
  function go(view) {
    if (VIEWS.indexOf(view) < 0) return;
    S.view = view;
    $$('.rail-btn[data-view]').forEach(b => {
      if (b.dataset.view === view) b.setAttribute('aria-current', 'page');
      else b.removeAttribute('aria-current');
    });
    $$('.view').forEach(v => v.classList.toggle('is-active', v.id === 'view-' + view));
    if (view === 'jobs') Jobs.render();
    if (view === 'analytics') {
      Analytics.render();
      // 任务汇总段依赖后端任务列表，而它不会自己定时刷新。
      // 进这个视图就拉一次；失败也只是那一段显示空状态，不影响其它三段。
      if (S.api.online) Jobs.refresh();
    }
    if (view === 'geo') Geo.render();
    if (view === 'settings') Settings.render();
    if (view === 'detect') {
      setTimeout(() => { Overlay.resize(); }, 30);
    }
    closePalette();
  }

  /**
   * 数据变化后重绘"当前正在显示"的视图。
   * 场景：用户先切到北斗/分析视图，此时帧还没加载完；等帧到位后如果不重绘，
   * 页面会一直停留在空状态。
   */
  function refreshView() {
    if (S.view === 'jobs') Jobs.render();
    else if (S.view === 'analytics') Analytics.render();
    else if (S.view === 'geo') Geo.render();
    else if (S.view === 'settings') Settings.render();
    else if (S.view === 'detect') renderStage();
  }

  /* ======================================================== 顶栏 / 状态栏 */
  function renderStatus() {
    const m = S.model, a = S.api;
    const pm = $('#pill-mode'), pd = $('#pill-device'), pv = $('#pill-model');

    let cls = 'chip', txt = '检测中…';
    if (!a.online) { cls = 'chip bad live'; txt = '未连接 · 本地演示'; }
    else if (m.mode === 'real') { cls = 'chip ok live'; txt = '真实推理'; }
    else { cls = 'chip warn live'; txt = '后端演示模式'; }
    pm.className = cls; pm.innerHTML = '<span class="dot"></span>' + txt;

    const dev = m.device === '0' || m.device === 'cuda' || /^\d/.test(m.device || '')
      ? (m.cudaName || 'GPU') : (m.device ? 'CPU' : '—');
    pd.className = 'chip' + (m.mode === 'real' && dev !== 'CPU' && dev !== '—' ? ' accent' : '');
    pd.innerHTML = '<span class="dot"></span>设备 ' + U.esc(dev);

    const wn = m.weightsLabel || (m.weights ? String(m.weights).split(/[\\/]/).pop() : (m.mode === 'real' ? '已加载' : '未加载'));
    pv.className = 'chip';
    pv.innerHTML = '<span class="dot"></span>模型 ' + U.esc(wn);

    // 状态栏
    const sc = $('#st-conn');
    sc.innerHTML = '<svg style="width:12px;height:12px"><use href="#' + (a.online ? 'i-plug' : 'i-plug-off') + '"/></svg>' +
      '<span>' + (a.online
        ? '已连接 · ' + (a.shape === 'fastapi' ? 'FastAPI' : 'Flask') + ' ' + a.prefix
        : '未连接后端 · 演示模式') + '</span>';
    sc.style.color = a.online ? 'var(--ok)' : 'var(--fg-3)';

    const sd = $('#st-device');
    sd.hidden = !a.online;
    if (a.online) sd.querySelector('span').textContent = dev;

    // 北斗
    if (S.beidou) Geo.renderTiles();
  }

  function setReqId(rid) {
    const el = $('#st-reqid');
    if (!rid) { el.hidden = true; return; }
    el.hidden = false;
    el.querySelector('span').textContent = 'req ' + String(rid).slice(0, 8);
  }

  function setLatency(ms) {
    const el = $('#st-ms');
    if (ms == null) { el.hidden = true; return; }
    el.hidden = false;
    el.querySelector('span').textContent = U.fmtMs(ms);
  }

  function showBanner(type, text) {
    const b = $('#banner');
    b.className = 'banner is-show ' + (type || '');
    $('#banner-t').textContent = text;
  }
  function hideBanner() { $('#banner').className = 'banner'; }

  /* ======================================================== 参数绑定 */
  function bindRange(el, valEl, key, fmt) {
    const upd = () => {
      const min = +el.min, max = +el.max;
      el.style.setProperty('--p', ((el.value - min) / (max - min) * 100) + '%');
      if (valEl) valEl.textContent = fmt ? fmt(el.value) : el.value;
    };
    // 用状态回填滑块：否则本地存储里的参数（如 conf=0.35）与 HTML 默认值（0.25）
    // 不一致时，滑块显示 0.25 而过滤实际按 0.35 走 —— 静默不符，极难排查。
    const cur = Store.getPath(key);
    if (cur != null && isFinite(+cur)) el.value = String(cur);
    el.addEventListener('input', () => {
      Store.setPath(key, +el.value, true);
      upd();
      if (key === 'params.conf') { renderRight(); renderStage(); }
      BDS.prefs.save();
    });
    upd();
  }

  function bindSeg(root, onPick, initial) {
    const btns = U.$$('.seg-btn', root);
    btns.forEach(b => {
      b.addEventListener('click', () => {
        btns.forEach(x => x.setAttribute('aria-selected', String(x === b)));
        onPick(b.dataset.v != null ? b.dataset.v : b.dataset.src);
      });
    });
    if (initial != null) {
      btns.forEach(b => b.setAttribute('aria-selected',
        String((b.dataset.v != null ? b.dataset.v : b.dataset.src) === String(initial))));
    }
  }

  function bindSwitch(el, key, onChange) {
    el.addEventListener('click', () => {
      const on = el.getAttribute('aria-checked') !== 'true';
      el.setAttribute('aria-checked', String(on));
      if (key) Store.setPath(key, on, true);
      if (onChange) onChange(on);
      BDS.prefs.save();
    });
  }
  function setSwitch(el, on) { el.setAttribute('aria-checked', String(!!on)); }

  /**
   * 把 S.params 里的推理参数**反向**刷回所有控件（滑块 / 分段 / 开关 / 数字框）。
   *
   * 起因：`params` 会持久化到 localStorage。用户把「置信度阈值」拖到 0.99 之后，
   * 页面每次打开都是"满屏 0"，而当时的「重置」按钮只清队列和选中，**不碰参数**——
   * 用户点了重置发现没变化，只会更确信"功能坏了"。
   * 所以既要有恢复默认的入口，也要有这条单向同步，保证状态与控件永远一致。
   */
  function syncParamUI() {
    const p = S.params;
    const set = (sel, v) => { const el = $(sel); if (el) el.value = v; };
    const txt = (sel, v) => { const el = $(sel); if (el) el.textContent = v; };
    const pct = (sel, v) => {
      const el = $(sel);
      if (!el) return;
      const min = +el.min, max = +el.max;
      el.style.setProperty('--p', ((v - min) / (max - min) * 100) + '%');
    };

    set('#p-conf', p.conf); txt('#p-conf-v', (+p.conf).toFixed(2)); pct('#p-conf', p.conf);
    set('#p-iou', p.iou); txt('#p-iou-v', (+p.iou).toFixed(2)); pct('#p-iou', p.iou);
    set('#p-maxdet', p.maxDet);
    set('#p-slice-size', p.sliceSize);
    set('#p-slice-overlap', p.sliceOverlap);
    pct('#p-slice-size', p.sliceSize); pct('#p-slice-overlap', p.sliceOverlap);
    txt('#p-slice-v', p.sliceSize + ' · ' + (+p.sliceOverlap).toFixed(2));
    U.$$('#seg-imgsz .seg-btn').forEach(b =>
      b.setAttribute('aria-selected', String(+b.dataset.v === +p.imgsz)));
    const sw = $('#p-slice');
    if (sw) { setSwitch(sw, p.slice); $('#p-slice-opts').hidden = !p.slice; }
    const sw2 = $('#set-slice');
    if (sw2) setSwitch(sw2, p.slice);
  }

  /* ======================================================== 队列与帧 */
  function addFiles(files) {
    const list = Array.prototype.slice.call(files).filter(f => /^image\//.test(f.type) || /\.(jpe?g|png|webp|bmp|gif)$/i.test(f.name));
    if (!list.length) { toast('warn', '没有可用的图片', '仅支持 JPG / PNG / WEBP / BMP'); return; }
    let n = 0;
    list.forEach(file => {
      const id = U.uid();
      const url = URL.createObjectURL(file);
      const item = { id: id, name: file.name, size: file.size, type: 'image', url: url, thumb: url, file: file };
      S.queue.push(item);
      S.frames[id] = {
        id: id, name: file.name, size: file.size, type: 'image', url: url, file: file,
        thumb: url, img: null, width: 0, height: 0, dets: [], done: false, demo: false,
      };
      U.loadImage(url).then(im => {
        const f = S.frames[id];
        f.img = im; f.width = im.width; f.height = im.height;
        if (S.activeId === id) showStage(f);
        renderQueue();
      });
      n++;
    });
    if (S.queue.length === 1) setActive(S.queue[0].id);
    else if (!S.activeId) setActive(S.queue[0].id);
    renderQueue();
    toast('ok', '已加入队列', n + ' 个文件 · 共 ' + S.queue.length + ' 项');
  }

  function removeItem(id) {
    S.queue = S.queue.filter(q => q.id !== id);
    delete S.frames[id];
    if (S.activeId === id) {
      const next = S.queue[0];
      if (next) setActive(next.id);
      else { S.activeId = null; clearStage(); }
    }
    renderQueue(); renderRight(); renderFilmstrip();
  }

  function clearQueue() {
    S.queue = []; S.frames = {}; S.activeId = null;
    clearStage(); renderQueue(); renderRight(); renderFilmstrip();
    toast('info', '队列已清空');
  }

  function setActive(id) {
    S.activeId = id;
    S.selected = -1;
    const f = S.frames[id];
    if (f) showStage(f);
    renderQueue(); renderFilmstrip(); renderRight();
    refreshView();
  }

  function activeFrame() { return S.activeId ? S.frames[S.activeId] : null; }

  function renderQueue() {
    const host = $('#filelist');
    $('#queue-count').textContent = S.queue.length;
    if (!S.queue.length) {
      host.innerHTML = '<div class="fld-hint" style="padding:6px 2px">队列为空。拖入图片或载入演示场景。</div>';
      return;
    }
    host.innerHTML = S.queue.map(q => {
      const f = S.frames[q.id] || {};
      const n = f.done ? (f.dets || []).length : null;
      return '<div class="fitem' + (q.id === S.activeId ? ' is-active' : '') + '" data-id="' + q.id + '">' +
        '<img class="thumb" src="' + q.thumb + '" alt="">' +
        '<span class="nm" title="' + U.esc(q.name) + '">' + U.esc(q.name) + '</span>' +
        (n != null ? '<span class="tag">' + n + '</span>' : '') +
        '<span class="sz">' + U.fmtBytes(q.size) + '</span>' +
        '<button class="ibtn sm x" data-rm="' + q.id + '" title="移除"><svg><use href="#i-x"/></svg></button>' +
        '</div>';
    }).join('');
    U.$$('.fitem', host).forEach(el => {
      el.onclick = e => {
        if (e.target.closest('[data-rm]')) return;
        setActive(el.dataset.id);
      };
    });
    U.$$('[data-rm]', host).forEach(b => {
      b.onclick = e => { e.stopPropagation(); removeItem(b.dataset.rm); };
    });
  }

  function renderFilmstrip() {
    const host = $('#filmstrip');
    const items = S.queue.filter(q => S.frames[q.id] && (S.frames[q.id].done || q.id === S.activeId));
    host.classList.toggle('is-empty', !items.length);
    if (!items.length) { host.innerHTML = ''; return; }
    host.innerHTML = items.map((q, i) => {
      const f = S.frames[q.id];
      return '<div class="ftile' + (q.id === S.activeId ? ' is-active' : '') + '" data-id="' + q.id + '" title="' + U.esc(q.name) + '">' +
        '<img src="' + (f.thumb || q.thumb) + '" alt="">' +
        (f.done ? '<span class="badge">' + f.dets.length + '</span>' : '') +
        '<span class="n">' + (i + 1) + '</span></div>';
    }).join('');
    U.$$('.ftile', host).forEach(el => { el.onclick = () => setActive(el.dataset.id); });
  }

  /* ======================================================== 舞台 */
  function clearStage() {
    Overlay.img = null; Overlay.dets = [];
    $('#cvwrap').classList.remove('is-visible');
    $('#stage-empty').style.display = '';
    $('#hud-tl').innerHTML = '<span class="k">STANDBY</span>';
    $('#hud-tr').hidden = true; $('#hud-bl').hidden = true; $('#hud-br').hidden = true;
    Overlay.resize();
  }

  function showStage(f) {
    if (!f || !f.img) { clearStage(); return; }
    $('#stage-empty').style.display = 'none';
    $('#cvwrap').classList.add('is-visible');
    Overlay.dets = f.dets || [];
    Overlay.opts = { visible: visibleClasses(), selected: S.selected, hover: S.hover };
    Overlay.setImage(f.img);
    if (f.type === 'video' || f.type === 'cam') Overlay.fit();
    Overlay.resize();
    renderStage();
    updateHUD(f);
  }

  function updateHUD(f) {
    const w = f.width || (f.img && f.img.width) || 0;
    const h = f.height || (f.img && f.img.height) || 0;
    $('#hud-tl').innerHTML = '<span class="k">' + U.esc((f.name || '').slice(0, 28)) + '</span>' +
      '<b>' + w + '×' + h + '</b>';
    const tr = $('#hud-tr');
    tr.hidden = false;
    tr.innerHTML = '<span class="k">引擎</span><b>' + U.esc(f.engine || (S.model.mode === 'real' ? 'yolo' : 'demo')) + '</b>' +
      '<span class="k">设备</span><b>' + U.esc(f.device === '0' ? 'GPU' : (f.device || '—')) + '</b>';
    const br = $('#hud-br');
    br.hidden = false;
    br.innerHTML = '<span class="k">耗时</span><b>' + U.fmtMs(f.ms) + '</b>' +
      '<span class="k">目标</span><b>' + filteredDets().length + '</b>';
  }

  function updateHUDCoords(sx, sy) {
    const f = activeFrame();
    if (!f) return;
    const p = Overlay.toImage(sx, sy);
    const bl = $('#hud-bl');
    bl.hidden = false;
    const g = BDS.geoFromPixel(p.x, p.y, f.width, f.height);
    bl.innerHTML = '<span class="k">x</span><b>' + Math.round(p.x) + '</b>' +
      '<span class="k">y</span><b>' + Math.round(p.y) + '</b>' +
      '<span class="k">LAT</span><b>' + (g ? g.lat : '—') + '</b>' +
      '<span class="k">LON</span><b>' + (g ? g.lon : '—') + '</b>';
  }

  function visibleClasses() {
    return CLASSES.filter(c => S.hidden.indexOf(c.id) < 0).map(c => c.id);
  }

  function renderStage() {
    const f = activeFrame();
    if (!f || !f.img) return;
    Overlay.dets = f.dets || [];
    Overlay.opts = {
      visible: visibleClasses(),
      selected: S.selected,
      hover: S.hover,
    };
    Overlay.render();
    $('#zoom-val').textContent = Math.round(Overlay.view.scale * 100) + '%';
  }

  /* ======================================================== 推理调度 */
  function setBusy(on, txt) {
    $('#busy').classList.toggle('is-show', !!on);
    if (txt) $('#busy-txt').textContent = txt;
  }

  async function runInference() {
    const f = activeFrame();
    if (!f) { toast('warn', '没有可推理的输入', '请先拖入图片或点击「载入演示场景」'); return; }
    if (S.running) return;
    if (!f.img && !f.file) { toast('warn', '图像尚未就绪', '请稍候再试'); return; }

    S.running = true;
    setBusy(true, S.api.online ? '推理中…' : '生成结果…');
    const t0 = performance.now();
    let ok = false;

    try {
      if (S.api.online) {
        const input = f.file ? f.file : (f.dataUrl || f.url);
        const r = await Api.detect(input, S.params, p => setBusy(true, '推理中… ' + p));
        setReqId(r.requestId);
        if (r.ok) {
          const d = r.data;
          if (d.beidou) applyBeidou(d.beidou);
          f.dets = d.detections;
          f.crowd = d.crowd || null;      // 人群密度（可能为 null）
          // 聚集预警优先于普通提示（更重要的信息不该被覆盖）
          if (d.crowd && d.crowd.level && d.crowd.level.alert) crowdAlert(d.crowd, d.beidou);
          else if (d.message) showBanner('warn', d.message);
          f.ms = d.elapsedMs != null ? d.elapsedMs : Math.round(performance.now() - t0);
          f.device = d.device; f.engine = d.engine;
          if (!f.img && d.original) {
            f.dataUrl = d.original;
            f.img = await U.loadImage(d.original);
            f.width = d.width || f.img.width;
            f.height = d.height || f.img.height;
          }
          f.done = true;
          ok = true;
        } else {
          toast('bad', '推理失败', r.error);
        }
      } else {
        if (f.demo && f.baked) {
          f.dets = f.baked.detections.map(d => Object.assign({}, d));
          f.ms = f.baked.ms;
          f.device = 'demo'; f.engine = 'demo';
          f.done = true; ok = true;
        } else {
          toast('warn', '后端未连接，无法推理',
            '当前为离线演示形态。点击「载入演示场景」可查看完整交互与可视化效果。');
        }
      }
    } catch (e) {
      toast('bad', '推理异常', String(e && e.message ? e.message : e));
    } finally {
      S.running = false;
      setBusy(false);
    }

    if (ok) {
      f.dets.forEach((d, i) => { d._i = i; });
      S.selected = -1;
      setLatency(f.ms);
      recordSession(f);
      showStage(f);
      renderRight(); renderFilmstrip();
      updateHUD(f);
      toast('ok', '推理完成', f.dets.length + ' 个目标 · ' + U.fmtMs(f.ms));
    }
    renderStage();
  }

  /* ======================================================== 演示场景 */
  let demoSeed = 20260913;
  async function loadDemo(newSeed) {
    if (newSeed) demoSeed = Math.floor(Math.random() * 1e9);
    setBusy(true, '生成演示场景…');
    await new Promise(r => setTimeout(r, 60));
    const baked = Demo.build(demoSeed);
    const id = U.uid();
    const thumb = await Demo.thumb(baked.dataUrl, 136);
    const img = await U.loadImage(baked.dataUrl);
    const name = '演示场景_' + String(demoSeed).slice(-4) + '.jpg';

    S.queue.push({ id: id, name: name, size: 0, type: 'demo', url: baked.dataUrl, thumb: thumb });
    S.frames[id] = {
      id: id, name: name, size: 0, type: 'demo', url: baked.dataUrl, thumb: thumb,
      img: img, width: baked.width, height: baked.height,
      dets: baked.detections.map(d => Object.assign({}, d)),
      ms: baked.ms, device: 'demo', engine: 'demo', done: true, demo: true, baked: baked,
    };
    S.frames[id].dets.forEach((d, i) => { d._i = i; });
    setBusy(false);
    setActive(id);
    S.activeId = id;
    recordSession(S.frames[id]);
    showStage(S.frames[id]);
    renderRight(); renderQueue(); renderFilmstrip();
    updateHUD(S.frames[id]);
    setLatency(baked.ms);
    showBanner('warn', S.api.online
      ? '演示场景：检测框为前端预置数据。点击「开始推理」可用后端真实模型对这张图重新推理。'
      : '演示场景：检测框为前端预置数据（未连接后端）。启动后端并刷新即可切换为真实推理。');
    toast('ok', '演示场景已载入', baked.detections.length + ' 个目标 · ' + baked.width + '×' + baked.height);
  }

  /* ======================================================== 会话统计 */
  function recordSession(f) {
    if (!f || !f.done) return;
    const s = S.session;
    s.frames++;
    const dets = f.dets || [];
    s.dets += dets.length;
    let sum = 0, small = 0;
    dets.forEach(d => {
      sum += d.conf;
      s.classCount[d.cls] = (s.classCount[d.cls] || 0) + 1;
      const bi = U.clamp(Math.floor(d.conf * 20), 0, 19);
      s.confHist[bi]++;
      if (U.sizeBucket(d.x2 - d.x1, d.y2 - d.y1) === 0) small++;
    });
    s.sumConf += sum; s.confCount += dets.length;
    s.smallDets += small; s.allDets += dets.length;
    s.latencies.push(f.ms || 0);
    if (s.latencies.length > 300) s.latencies.shift();
    s.history.unshift({ t: Date.now(), n: dets.length, ms: f.ms || 0, avg: dets.length ? sum / dets.length : 0, name: f.name });
    if (s.history.length > 60) s.history.pop();
    saveSession();
  }

  const SESSION_KEY = 'bds.session.v1';
  function saveSession() {
    if (!S.persist) return;
    try { localStorage.setItem(SESSION_KEY, JSON.stringify(S.session)); } catch (e) { /* ignore */ }
  }
  function loadSession() {
    try {
      const raw = localStorage.getItem(SESSION_KEY);
      if (!raw) return;
      const s = JSON.parse(raw);
      if (s && typeof s.frames === 'number') {
        Object.assign(S.session, s);
        if (!Array.isArray(S.session.confHist) || S.session.confHist.length !== 20) S.session.confHist = new Array(20).fill(0);
        if (!Array.isArray(S.session.classCount) || S.session.classCount.length !== 10) S.session.classCount = new Array(10).fill(0);
      }
    } catch (e) { /* ignore */ }
  }
  function clearSession() {
    S.session = {
      frames: 0, dets: 0, sumConf: 0, confCount: 0, latencies: [],
      classCount: new Array(10).fill(0), confHist: new Array(20).fill(0),
      history: [], smallDets: 0, allDets: 0,
    };
    saveSession();
  }

  /**
   * 导出本机累计的分析数据。设置页与分析视图各有一个入口，
   * 逻辑只写一份 —— 两处各写一遍迟早会漏字段（导出的是"分析结果"，
   * 少一个字段就等于导出了一份错的）。
   */
  function exportSession() {
    U.download('session_' + Date.now() + '.json',
      JSON.stringify({ session: S.session, params: S.params, beidou: S.beidou }, null, 2),
      'application/json');
    toast('ok', '已导出会话数据');
  }

  /* ======================================================== 右栏 */
  function filteredDets() {
    const f = activeFrame();
    if (!f || !f.dets) return [];
    const conf = S.params.conf;
    const hidden = S.hidden;
    let q = S.search.trim();
    let minC = conf;
    let text = '';
    if (q) {
      const m = q.match(/^([<>]=?|min:)\s*([0-9.]+)$/);
      if (m) {
        const v = parseFloat(m[2]);
        if (m[1] === '>') minC = Math.max(minC, v);
        else if (m[1] === '>=' || m[1] === 'min:') minC = Math.max(minC, v);
        else if (m[1] === '<') { minC = Math.max(minC, 0); }
      } else {
        text = q.toLowerCase();
      }
    }
    let arr = f.dets.filter(d => d.conf >= minC && hidden.indexOf(d.cls) < 0);
    if (text) {
      arr = arr.filter(d => {
        const c = CLASSES[d.cls] || {};
        return (c.zh || '').toLowerCase().indexOf(text) >= 0 || (c.en || '').toLowerCase().indexOf(text) >= 0;
      });
    }
    const s = S.sort;
    arr = arr.slice();
    if (s === 'conf') arr.sort((a, b) => b.conf - a.conf);
    else if (s === 'confAsc') arr.sort((a, b) => a.conf - b.conf);
    else if (s === 'class') arr.sort((a, b) => a.cls - b.cls || b.conf - a.conf);
    else if (s === 'area') arr.sort((a, b) => (b.x2 - b.x1) * (b.y2 - b.y1) - (a.x2 - a.x1) * (a.y2 - a.y1));
    return arr;
  }

  /* ======================================================== 人群密度面板 */
  // 分级配色：正常绿 / 关注橙 / 警戒橙 / 危险红
  // ⚠ 值必须是 app.css 里真实存在的 chip 修饰类：.ok / .warn / .bad / .accent / .geo。
  //   写 'danger' 不会报错，只会静默退化成无色 chip —— 危险等级看不出来。
  const CROWD_COLOR = { normal: 'ok', watch: 'warn', warn: 'warn', danger: 'bad' };

  function renderCrowd(f) {
    const c = f && f.crowd;
    const el = {
      lv: $('#crowd-level'), cnt: $('#crowd-count'),
      den: $('#crowd-density'), occ: $('#crowd-occ'), brk: $('#crowd-breakdown'),
      th: $('#crowd-th'), hint: $('#crowd-hint'),
    };
    if (!el.lv) return;

    if (!c) {
      el.lv.textContent = '—'; el.lv.className = 'tag';
      el.cnt.textContent = '—'; el.den.textContent = '—';
      el.occ.textContent = '—'; el.brk.textContent = '—';
      if (el.th) el.th.textContent = '—';
      if (el.hint) el.hint.textContent =
        '未标定：只统计人数，不给密度。填入画面实际覆盖面积后重跑，即可得出密度与分级。';
      renderVideoCrowdTip();
      return;
    }

    el.cnt.textContent = c.count + ' 人';
    el.occ.textContent = (c.occupancy_ratio * 100).toFixed(2) + '%';
    el.brk.textContent = '行人 ' + c.count_pedestrian + ' · 人群 ' + c.count_people +
      (c.people_weight !== 1 ? '（人群框 ×' + c.people_weight + '）' : '');

    const th = c.thresholds || {};
    if (el.th) {
      el.th.textContent = [th.watch, th.warn, th.danger].every(v => v != null)
        ? th.watch + ' / ' + th.warn + ' / ' + th.danger + ' 人/m²'
        : '—';
    }

    if (c.calibrated && c.level) {
      el.den.textContent = c.density + ' 人/m²';
      // 用现成的语义色 chip，避免自己造样式
      el.lv.className = 'chip ' + (CROWD_COLOR[c.level.key] || '') + (c.level.alert ? ' live' : '');
      el.lv.innerHTML = '<span class="dot"></span>' + c.level.zh;
      if (el.hint) {
        el.hint.textContent = '已按 ' + c.area_m2 + ' m² 标定。'
          + (c.level.alert ? '当前已触发聚集预警。' : '人群越密遮挡越重，高密度下计数会系统性偏低。');
      }
    } else {
      el.den.textContent = '未标定';
      el.lv.className = 'tag';
      el.lv.textContent = '未标定';
      if (el.hint) {
        el.hint.textContent = c.message ||
          '未标定：只统计人数，不给密度。填入画面实际覆盖面积后重跑，即可得出密度与分级。';
      }
    }
    renderVideoCrowdTip();
  }

  /** 视频页签下的标定状态提示：不标定时视频只有人数、没有密度曲线，必须提前说清 */
  function renderVideoCrowdTip() {
    const el = $('#video-crowd-tip');
    if (!el) return;
    const a = S.params.areaM2;
    if (a == null) {
      el.innerHTML = '人群密度：<b>未标定</b> —— 视频结果只给逐帧人数，没有密度曲线与分级。'
        + '在右侧「人群密度」面板填入画面实际覆盖面积即可。';
    } else {
      el.innerHTML = '人群密度：已标定 <b>' + a + ' m²</b> —— 视频结果会给出逐帧密度曲线与聚集预警。';
    }
  }

  /** 触发聚集预警横幅（带北斗坐标） */
  function crowdAlert(c, beidou) {
    if (!c || !c.level || !c.level.alert) return;
    let msg = '人群聚集预警：' + c.count + ' 人 · 密度 ' + c.density +
      ' 人/m²（' + c.level.zh + '）';
    if (beidou && beidou.lat != null) {
      msg += ' · 位置 ' + (+beidou.lat).toFixed(5) + ', ' + (+beidou.lon).toFixed(5);
    }
    showBanner('bad', msg);
  }

  /**
   * 右侧被清空时，说清楚「为什么一个都没有」。
   *
   * 起因：置信度阈值滑块拉到 0.99（且会持久化到 localStorage），于是所有目标被过滤，
   * 界面上只剩一排 0 —— 看起来像功能坏了，实际是自己把阈值拉满了。
   * 这种时候给出原因，比再画十根空条有用得多。
   */
  function emptyReason() {
    const f = activeFrame();
    const all = f ? (f.dets || []) : [];
    const total = all.length;
    const conf = S.params.conf;
    if (!total) return '尚未推理，或这张图没有检出任何目标。';
    if (S.hidden.length >= CLASSES.length) {
      return '全部 ' + CLASSES.length + ' 个类别都被手动隐藏了（共 ' + total + ' 个目标）。点击任一类别即可恢复。';
    }
    if (!all.some(d => d.conf >= conf)) {
      return '置信度阈值 <b>' + conf + '</b> 过高：' + total +
        ' 个候选框被全部过滤。把左侧「置信度阈值」调低到 0.25 左右，' +
        '或点「推理参数」右上角的恢复按钮一键还原。';
    }
    if (S.hidden.length) return '共 ' + total + ' 个目标，其中 ' + S.hidden.length + ' 个类别被隐藏。';
    if (S.search.trim()) return '没有匹配「' + U.esc(S.search.trim()) + '」的目标。';
    return '共 ' + total + ' 个目标，均未通过当前筛选条件。';
  }

  function renderRight() {
    const f = activeFrame();
    const all = f ? (f.dets || []) : [];
    const dets = filteredDets();                    // 已应用：阈值 + 类别隐藏 + 搜索 + 排序
    const confSet = all.filter(d => d.conf >= S.params.conf);  // 仅按阈值过滤（用于类别分布）

    // KPI —— 以"当前可见"为准，与列表口径一致
    const vis = dets.length;
    $('#kpi-total').textContent = vis;
    $('#kpi-total-d').textContent = all.length > vis
      ? '共 ' + all.length + ' 个 · 已过滤 ' + (all.length - vis)
      : (f ? (f.width + '×' + f.height) : '—');
    const avg = vis ? dets.reduce((a, d) => a + d.conf, 0) / vis : null;
    $('#kpi-conf').textContent = avg == null ? '—' : (avg * 100).toFixed(1) + '%';
    $('#kpi-conf-d').textContent = vis ? '最高 ' + (Math.max.apply(null, dets.map(d => d.conf)) * 100).toFixed(1) + '%' : '—';
    $('#kpi-ms').innerHTML = (f && f.ms != null) ? Math.round(f.ms) + '<small>ms</small>' : '—';
    $('#kpi-ms-d').textContent = f && f.ms != null ? (all.length / Math.max(1, f.ms) * 1000).toFixed(1) + ' obj/s' : '—';
    const smallN = dets.filter(d => U.sizeBucket(d.x2 - d.x1, d.y2 - d.y1) === 0).length;
    $('#kpi-small').textContent = vis ? Math.round(smallN / vis * 100) + '%' : '—';
    $('#kpi-small-d').textContent = vis ? smallN + ' / ' + vis + ' 个 < 32² px' : '< 32² px';

    renderCrowd(f);   // 人群密度面板

    // 类别分布（口径：通过阈值的目标，包含被隐藏的类别，便于重新勾选）
    // 只给数字，不画比例条：条形的相对长度容易被误读成"没有数据"，
    // 而且一排 0 的时候真正该说的是"为什么是 0"，不是再画十根空条。
    const counts = CLASSES.map(c => confSet.filter(d => d.cls === c.id).length);
    $('#dist-count').textContent = confSet.length;
    $('#dist').innerHTML = confSet.length
      ? CLASSES.map((c, i) => {
        const off = S.hidden.indexOf(c.id) >= 0;
        return '<div class="dist-row' + (off ? ' is-off' : '') + (counts[i] ? '' : ' is-zero') +
          '" data-cls="' + c.id + '" title="点击隐藏/显示该类别">' +
          '<i class="dot" style="background:' + BDS.classColor(c.id) + '"></i>' +
          '<span class="nm">' + U.esc(c.zh) + '</span>' +
          '<span class="sp"></span>' +
          '<span class="n">' + counts[i] + '</span></div>';
      }).join('')
      : '<div class="dist-hint">' + emptyReason() + '</div>';
    U.$$('#dist .dist-row').forEach(el => {
      el.onclick = () => {
        const id = +el.dataset.cls;
        const k = S.hidden.indexOf(id);
        if (k >= 0) S.hidden.splice(k, 1); else S.hidden.push(id);
        renderRight(); renderStage();
      };
    });

    // 列表
    const CAP = 400;
    $('#det-count').textContent = dets.length;
    const host = $('#det-list');
    if (!dets.length) {
      host.innerHTML = '<div class="empty"><div class="empty-ic"><svg><use href="#i-dashed"/></svg></div>' +
        '<h4>' + (all.length ? '当前筛选无结果' : '暂无检测结果') + '</h4>' +
        '<p>' + (all.length ? emptyReason() : '载入图像并执行推理后，目标列表会出现在这里。') + '</p></div>';
    } else {
      host.innerHTML = dets.slice(0, CAP).map(d => {
        const c = CLASSES[d.cls] || { zh: '?', en: '' };
        const col = BDS.classColor(d.cls);
        const w = Math.round(d.x2 - d.x1), h = Math.round(d.y2 - d.y1);
        const sel = S.selected === d._i;
        return '<div class="det-row' + (sel ? ' is-sel' : '') + '" data-i="' + d._i + '" style="--c:' + col + '">' +
          '<i class="sw"></i>' +
          '<div class="main"><div class="l1"><span class="zh">' + U.esc(c.zh) + '</span>' +
          '<span class="en">' + U.esc(c.en) + '</span></div>' +
          '<div class="l2"><span>' + w + '×' + h + ' px</span>' +
          (U.sizeBucket(w, h) === 0 ? '<span class="tag" style="height:14px">小</span>' : '') +
          (d.lon != null ? '<span>' + d.lat.toFixed(4) + ', ' + d.lon.toFixed(4) + '</span>' : '') +
          '</div></div>' +
          '<div class="cf-bar"><i style="width:' + (d.conf * 100).toFixed(0) + '%"></i></div>' +
          '<span class="cf">' + (d.conf * 100).toFixed(1) + '%</span>' +
          '</div>';
      }).join('') + (dets.length > CAP
        ? '<div class="fld-hint" style="padding:8px 12px">仅显示前 ' + CAP + ' 条（共 ' + dets.length + ' 条）。</div>' : '');
      U.$$('#det-list .det-row').forEach(el => {
        el.onclick = () => selectDet(+el.dataset.i);
        el.onmouseenter = () => { S.hover = +el.dataset.i; renderStage(); };
        el.onmouseleave = () => { S.hover = -1; renderStage(); };
      });
    }

    renderDetail();
  }

  function renderDetail() {
    const box = $('#detail');
    const f = activeFrame();
    if (!f || S.selected < 0 || !f.dets[S.selected]) { box.hidden = true; return; }
    const d = f.dets[S.selected];
    const c = CLASSES[d.cls] || { zh: '?', en: '' };
    const col = BDS.classColor(d.cls);
    const w = d.x2 - d.x1, h = d.y2 - d.y1;
    const g = (d.lat != null && d.lon != null) ? { lat: d.lat, lon: d.lon } : BDS.geoFromPixel((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2, f.width, f.height);
    const enu = BDS.enuFromPixel((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2, f.width, f.height);
    box.hidden = false;
    $('#detail-body').innerHTML =
      '<div class="dh"><i class="dot" style="background:' + col + '"></i>' +
      '<span class="zh">' + U.esc(c.zh) + '</span>' +
      '<span class="tag">' + U.esc(c.en) + '</span>' +
      '<div class="sp"></div><span class="cf mono" style="color:var(--fg-2)">' + (d.conf * 100).toFixed(1) + '%</span></div>' +
      '<dl class="kv kv2">' +
      '<dt style="grid-column:1">边界框</dt>' +
      '<dd style="grid-column:2 / -1">' + d.x1.toFixed(0) + ', ' + d.y1.toFixed(0) +
      '  →  ' + d.x2.toFixed(0) + ', ' + d.y2.toFixed(0) + '</dd>' +
      '<dt>尺寸</dt><dd>' + w.toFixed(0) + ' × ' + h.toFixed(0) + ' px</dd>' +
      '<dt>目标尺度</dt><dd>' + ['小目标', '中目标', '大目标'][U.sizeBucket(w, h)] + '</dd>' +
      '<dt>面积</dt><dd>' + U.fmtNum(w * h) + ' px²</dd>' +
      '<dt>中心像素</dt><dd>' + ((d.x1 + d.x2) / 2).toFixed(0) + ', ' + ((d.y1 + d.y2) / 2).toFixed(0) + '</dd>' +
      '<dt>纬度 LAT</dt><dd>' + g.lat + '</dd>' +
      '<dt>经度 LON</dt><dd>' + g.lon + '</dd>' +
      '<dt>本地东向</dt><dd>' + (enu.e >= 0 ? '+' : '') + enu.e + ' m</dd>' +
      '<dt>本地北向</dt><dd>' + (enu.n >= 0 ? '+' : '') + enu.n + ' m</dd>' +
      '</dl>';
  }

  function selectDet(i) {
    S.selected = (S.selected === i ? -1 : i);
    renderRight(); renderStage();
    if (S.selected >= 0) {
      const row = $('#det-list .det-row[data-i="' + S.selected + '"]');
      if (row) row.scrollIntoView({ block: 'nearest' });
    }
  }

  /* ======================================================== 北斗 */
  function applyBeidou(b) {
    // 用默认帧兜底而不是整体替换：早期后端只返回 lat/lon/alt/satellites/real 五个字段，
    // 整体替换会让 usable / source / gsd_m_per_px 变成 undefined，于是界面上出现
    // "undefined"，或者被 Number() 当成 0 —— 那就是一个假坐标。
    S.beidou = Object.assign(JSON.parse(JSON.stringify(BDS.DEFAULTS.beidou)), b || {});
    Geo.renderTiles();
    // 星空图也要跟着换：只刷瓦片会让「chip 显示未接入定位源」和
    // 「星空图标着 GSV 实测 · 12 颗」同时出现在一屏上，自相矛盾。
    if (S.view === 'geo') Geo.renderSky();
  }

  /* ======================================================== 导出 */
  function exportResults(format) {
    const f = activeFrame();
    if (!f || !f.done) { toast('warn', '没有可导出的结果', '请先执行一次推理'); return; }
    const dets = f.dets || [];
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

    if (format === 'csv') {
      const rows = ['index,class_id,class_zh,class_en,confidence,x1,y1,x2,y2,width,height,lat,lon'];
      dets.forEach((d, i) => {
        const c = CLASSES[d.cls] || {};
        rows.push([i + 1, d.cls, c.zh, c.en, d.conf, d.x1, d.y1, d.x2, d.y2,
          (d.x2 - d.x1).toFixed(1), (d.y2 - d.y1).toFixed(1),
          d.lat != null ? d.lat : '', d.lon != null ? d.lon : ''].join(','));
      });
      U.download('detections_' + stamp + '.csv', '\ufeff' + rows.join('\n'), 'text/csv;charset=utf-8');
      toast('ok', '已导出 CSV', dets.length + ' 条记录');
      return;
    }

    if (format === 'png') {
      const cv = document.createElement('canvas');
      const scale = Math.min(1, 2400 / Math.max(f.width, 1));
      cv.width = Math.round(f.width * scale);
      cv.height = Math.round(f.height * scale);
      const g = cv.getContext('2d');
      g.drawImage(f.img, 0, 0, cv.width, cv.height);
      const visible = visibleClasses();
      dets.forEach((d, i) => {
        if (visible.indexOf(d.cls) < 0) return;
        const col = BDS.classColor(d.cls);
        const x = d.x1 * scale, y = d.y1 * scale, w = (d.x2 - d.x1) * scale, h = (d.y2 - d.y1) * scale;
        g.lineWidth = Math.max(1.5, 2 * scale);
        g.strokeStyle = col;
        g.strokeRect(x, y, w, h);
        const label = (CLASSES[d.cls] || {}).zh + ' ' + (d.conf * 100).toFixed(0) + '%';
        g.font = '600 ' + Math.max(12, 14 * scale) + 'px ' + getComputedStyle(document.body).fontFamily;
        const tw = g.measureText(label).width + 10;
        const th = Math.max(18, 20 * scale);
        g.fillStyle = col; g.fillRect(x, y - th, tw, th);
        g.fillStyle = '#08101a';
        g.textBaseline = 'middle';
        g.fillText(label, x + 5, y - th / 2);
      });
      cv.toBlob(b => {
        U.download('annotated_' + stamp + '.png', b);
        toast('ok', '已导出标注图', cv.width + '×' + cv.height);
      }, 'image/png');
      return;
    }

    // JSON
    const payload = {
      meta: {
        system: '北斗 · 改进 YOLOv8m 小目标识别辅助系统',
        exported_at: new Date().toISOString(),
        source: f.name,
        width: f.width, height: f.height,
        mode: S.model.mode, device: f.device || null, engine: f.engine || null,
        elapsed_ms: f.ms != null ? f.ms : null,
        params: S.params,
      },
      beidou: S.beidou,
      classes: CLASSES,
      detections: dets.map((d, i) => ({
        index: i + 1,
        class_id: d.cls,
        name_zh: (CLASSES[d.cls] || {}).zh,
        name_en: (CLASSES[d.cls] || {}).en,
        confidence: +d.conf.toFixed(4),
        bbox: { x1: +d.x1.toFixed(2), y1: +d.y1.toFixed(2), x2: +d.x2.toFixed(2), y2: +d.y2.toFixed(2) },
        lon: d.lon, lat: d.lat,
      })),
    };
    U.download('detections_' + stamp + '.json', JSON.stringify(payload, null, 2), 'application/json');
    toast('ok', '已导出 JSON', dets.length + ' 条记录');
  }

  /* ======================================================== 任务视图 */
  /**
   * 视频任务详情里的逐帧人群曲线。
   * 人数（0~几十）和密度（0~几）量纲差太远，挤在一根轴上密度会被压成直线，
   * 所以拆成两张图；密度图额外叠加 warn / danger 阈值参考线。
   */
  function renderCrowdCharts(stats) {
    const cr = stats && stats.crowd;
    if (!cr || !cr.frames) return;

    const c1 = $('#crowd-chart-count');
    if (c1 && (cr.series_count || []).length) {
      Charts.curve(c1, [{ name: '每帧人数', values: cr.series_count }], {
        height: 182, yFloor0: true, xStart: 1,
        bestPrefix: '峰值 ', bestSuffix: 'f',
        fmt: v => String(Math.round(v)), empty: '无逐帧人数数据',
      });
    }

    const c2 = $('#crowd-chart-density');
    if (c2 && cr.calibrated && (cr.series_density || []).length) {
      const th = cr.thresholds || {};
      const hlines = [];
      if (isFinite(th.warn)) hlines.push({ v: th.warn, color: 'var(--warn)', label: '警戒 ' + th.warn });
      if (isFinite(th.danger)) hlines.push({ v: th.danger, color: 'var(--danger)', label: '危险 ' + th.danger });
      Charts.curve(c2, [{ name: '密度 (人/m²)', values: cr.series_density, color: 'var(--warn)' }], {
        height: 182, yFloor0: true, xStart: 1, hlines: hlines,
        bestPrefix: '峰值 ', bestSuffix: 'f',
        fmt: v => Number(v).toFixed(2), empty: '无逐帧密度数据',
      });
    }
  }

  const Jobs = {
    render() {
      const host = $('#job-list');
      $('#job-count').textContent = S.jobs.length;
      if (!S.jobs.length) {
        host.innerHTML = '<div class="empty"><div class="empty-ic"><svg><use href="#i-film"/></svg></div>' +
          '<h4>暂无推理任务</h4><p>在「检测台」选择视频文件提交后，任务会出现在这里。视频逐帧推理在服务端异步执行。</p>' +
          '<button class="btn btn-sm" id="btn-job-go"><svg><use href="#i-scan"/></svg>前往检测台</button></div>';
        const b = $('#btn-job-go');
        if (b) b.onclick = () => go('detect');
        Jobs.renderDetail();
        return;
      }
      host.innerHTML = '<div class="job-head"><span></span><span>文件</span>' +
        '<span>状态</span><span>目标</span><span>大小</span></div>' +
        S.jobs.map(j => {
          const st = j.status || 'queued';
          const icon = st === 'completed' ? 'i-check' : st === 'failed' ? 'i-alert' : 'i-clock';
          const pr = j.progress || { done: 0, total: 0 };
          const pct = pr.total ? Math.round(pr.done / pr.total * 100) : 0;
          const n = (j.result && j.result.stats) ? j.result.stats.total : (j.local && j.localCount != null ? j.localCount : null);
          return '<div class="job-row' + (j.id === Jobs.sel ? ' is-sel' : '') + '" data-id="' + U.esc(j.id) + '">' +
            '<div class="job-ic ' + st + '"><svg><use href="#' + icon + '"/></svg></div>' +
            '<div class="jt"><div class="t1">' + U.esc(j.name || ('任务 ' + String(j.id).slice(0, 8))) + '</div>' +
            '<div class="t2">' + U.esc(String(j.id).slice(0, 12)) + ' · ' + U.ago(j.createdAt || j.created_at) + '</div></div>' +
            '<div class="num">' + (st === 'processing' ? pct + '%' : (st === 'completed' ? '完成' : st === 'failed' ? '失败' : '排队')) + '</div>' +
            '<div class="num">' + (n != null ? n + ' obj' : '—') + '</div>' +
            '<div class="num">' + (j.size ? U.fmtBytes(j.size) : '—') + '</div>' +
            // 进行中：行底一条细进度条，比只给一个百分比直观
            (st === 'processing' && pr.total
              ? '<div class="jbar"><i style="width:' + pct + '%"></i></div>' : '') +
            '</div>';
        }).join('');
      U.$$('#job-list .job-row').forEach(el => {
        el.onclick = () => { Jobs.sel = el.dataset.id; Jobs.render(); };
      });
      Jobs.renderDetail();
    },

    renderDetail() {
      const host = $('#job-detail');
      const chip = $('#job-status-chip');
      const j = S.jobs.find(x => x.id === Jobs.sel);
      if (!j) {
        chip.className = 'chip'; chip.innerHTML = '<span class="dot"></span>—';
        host.innerHTML = '<div class="empty"><div class="empty-ic"><svg><use href="#i-dashed"/></svg></div>' +
          '<h4>未选择任务</h4><p>从左侧列表选择一个任务查看详情。</p></div>';
        return;
      }
      const st = j.status || 'queued';
      const map = { queued: ['chip', '排队中'], processing: ['chip accent live', '处理中'], completed: ['chip ok', '已完成'], failed: ['chip bad', '失败'] };
      const mc = map[st] || map.queued;
      chip.className = mc[0]; chip.innerHTML = '<span class="dot"></span>' + mc[1];

      const pr = j.progress || { done: 0, total: 0 };
      const stats = (j.result && j.result.stats) || null;
      let html = '';
      html += '<div class="sec"><div class="sec-hd"><span class="t">进度</span><div class="sp"></div>' +
        '<span class="tag">' + pr.done + ' / ' + (pr.total || '?') + '</span></div>' +
        '<div class="sec-bd"><div class="progress"><i style="width:' + (pr.total ? (pr.done / pr.total * 100) : (st === 'completed' ? 100 : 0)) + '%"></i></div></div></div>';

      html += '<div class="sec"><div class="sec-hd"><span class="t">任务信息</span></div><div class="sec-bd">' +
        '<dl class="kv kv-l">' +
        '<dt>任务 ID</dt><dd>' + U.esc(j.id) + '</dd>' +
        '<dt>文件</dt><dd>' + U.esc(j.name || '—') + '</dd>' +
        '<dt>大小</dt><dd>' + (j.size ? U.fmtBytes(j.size) : '—') + '</dd>' +
        '<dt>创建</dt><dd>' + U.fmtDate(j.createdAt || j.created_at) + '</dd>' +
        '<dt>参数</dt><dd>conf ' + (j.params && j.params.conf != null ? j.params.conf : S.params.conf) +
        ' · iou ' + (j.params && j.params.iou != null ? j.params.iou : S.params.iou) + '</dd>' +
        '</dl></div></div>';

      if (j.error) {
        html += '<div class="sec"><div class="sec-hd"><span class="t">错误</span></div>' +
          '<div class="sec-bd"><div class="banner is-show bad" style="position:static;transform:none;max-width:none">' +
          '<svg><use href="#i-alert"/></svg><span>' + U.esc(j.error) + '</span></div></div></div>';
      }

      if (stats) {
        html += '<div class="sec"><div class="sec-hd"><span class="t">统计</span></div><div class="sec-bd">' +
          '<dl class="kv kv-l">' +
          '<dt>总目标数</dt><dd>' + stats.total + '</dd>' +
          '<dt>帧数</dt><dd>' + stats.frames + '</dd>' +
          '<dt>平均每帧</dt><dd>' + stats.avg_per_frame + '</dd>' +
          '<dt>峰值每帧</dt><dd>' + stats.max_per_frame + '</dd>' +
          '</dl></div></div>';

        // ---- 人群密度与聚集预警（后端 stats.crowd，逐帧累积后的汇总）----
        const cr = stats.crowd;
        if (cr && cr.frames) {
          const lv = cr.peak_level || { key: 'normal', zh: '—', alert: false };
          const th = cr.thresholds || {};
          const pct = ((cr.alert_ratio || 0) * 100).toFixed(1);
          let cd = '<dt>峰值人数</dt><dd>' + cr.peak_count + ' 人</dd>' +
            '<dt>平均人数</dt><dd>' + cr.avg_count + ' 人</dd>';
          if (cr.calibrated) {
            cd += '<dt>峰值密度</dt><dd>' + cr.peak_density + ' 人/m²</dd>' +
              '<dt>平均密度</dt><dd>' + (cr.avg_density != null ? cr.avg_density : '—') + ' 人/m²</dd>';
          } else {
            cd += '<dt>密度</dt><dd>未标定 —— 只有人数，没有密度与分级</dd>';
          }
          cd += '<dt>预警帧数</dt><dd>' + cr.alert_frames + ' / ' + cr.frames + ' 帧（' + pct + '%）</dd>' +
            '<dt>分级阈值</dt><dd>' + (th.watch != null ? th.watch : '—') + ' / ' +
            (th.warn != null ? th.warn : '—') + ' / ' +
            (th.danger != null ? th.danger : '—') + ' 人/m²</dd>';

          html += '<div class="sec"><div class="sec-hd"><span class="t">人群密度与聚集预警</span><div class="sp"></div>' +
            '<span class="chip ' + (CROWD_COLOR[lv.key] || '') + (lv.alert ? ' live' : '') + '">' +
            '<span class="dot"></span>峰值 ' + U.esc(lv.zh) + '</span></div>' +
            '<div class="sec-bd">' +
            // 单靠一个 chip 太容易漏看，达到预警级别直接给一条横幅
            (lv.alert
              ? '<div class="banner is-show bad" style="position:static;transform:none;max-width:none;margin-bottom:10px">' +
                '<svg><use href="#i-alert"/></svg><span>检测到人群聚集：峰值 <b>' + cr.peak_count +
                ' 人</b> · 密度 <b>' + cr.peak_density + ' 人/m²</b>（' + U.esc(lv.zh) +
                '）· 全片 ' + cr.alert_frames + ' / ' + cr.frames + ' 帧越线（' + pct + '%）。' +
                '</span></div>'
              : '') +
            '<dl class="kv kv-l">' + cd + '</dl>' +
            '<div id="crowd-chart-count" style="margin-top:12px"></div>' +
            (cr.calibrated ? '<div id="crowd-chart-density" style="margin-top:4px"></div>' : '') +
            (cr.calibrated ? '' :
              '<div class="fld-hint" style="margin-top:10px">在「检测」页右侧面板填写画面实际覆盖面积并应用，' +
              '再跑一次视频即可得到密度曲线与自动分级。</div>') +
            '</div></div>';
        }

        // 检出 0 个时结果视频必然"和原片一模一样"（只是重新编码了一遍），
        // 很容易被误判成"标注没生效"。这里显式说明原因，别让用户白排查。
        if (!stats.total && st === 'completed') {
          html += '<div class="sec"><div class="sec-bd">' +
            '<div class="banner is-show warn" style="position:static;transform:none;max-width:none">' +
            '<svg><use href="#i-alert"/></svg><span>' +
            '全流程跑通了，但这段视频<b>一个目标都没检出</b>，所以结果视频与原片看起来完全一样' +
            '（只是重新编码，没有框可画）。<br>' +
            '本模型用 VisDrone <b>无人机航拍</b>数据训练，只认行人 / 轿车 / 巴士 / 卡车等航拍视角目标；' +
            '屏幕录制、室内监控、普通照片这类非航拍素材通常检不出东西。' +
            '建议换一段无人机航拍视频再试，或把置信度阈值调低到 0.15 左右看看。' +
            '</span></div></div></div>';
        }
      }

      if (j.result && j.result.video_url) {
        const src = Api.asset(j.result.video_url);
        html += '<div class="sec"><div class="sec-hd"><span class="t">结果视频</span><div class="sp"></div>' +
          '<a class="btn btn-sm" href="' + src + '" download><svg><use href="#i-download"/></svg>下载</a></div>' +
          '<div class="sec-bd"><video src="' + src + '" controls style="width:100%;border-radius:9px;background:#000"></video>' +
          '<div class="job-fix" id="job-fix"></div></div></div>';
      }

      host.innerHTML = html;
      renderCrowdCharts(stats);
      Jobs._bindVideoFix(host);
    },

    /**
     * 让结果视频的进度条带动定位读数。
     *
     * 用户的实际用法就是"视频和存储卡内容一起拖进来，拖动看某一时刻在哪"，
     * 所以拖动视频时这里必须跟着变 —— 而不是让人对着两个时间轴手动换算。
     *
     * 取帧走 :meth:`Track.frameAt` 的**本地插值**，因此可以放开到每帧一次：
     * 原先按 5 Hz 节流是因为每次都发 HTTP，而拖动进度条时请求排队会把
     * 浏览器的连接池拖死。本地化之后既没有请求、读数也能贴着画面走。
     */
    _bindVideoFix(host) {
      const vid = host.querySelector('video');
      const box = host.querySelector('#job-fix');
      if (!vid || !box) return;
      if (!S.track || !S.track.loaded) {
        box.innerHTML = '<p class="fld-hint" style="padding:8px 0 0">' +
          '想在这里看到每一刻的位置，请到「北斗」视图导入存储卡日志。</p>';
        return;
      }
      const sm = S.track.summary || {};
      const meta = '<p class="fld-hint" style="padding:0 0 8px">' +
        '视频 ' + (vid.duration ? vid.duration.toFixed(1) : '—') + ' s · 日志 ' +
        (+sm.duration_s || 0).toFixed(1) + ' s' +
        (sm.aligned ? '' : ' · <b>尚未对齐</b>（按起点回放，可在「北斗」视图校正）') +
        '</p>';
      // 整块重绘（元信息 + 读数）而不是两个事件各写一半：
      // 分开写会让 loadedmetadata 的提示被随后到来的 timeupdate 覆盖，
      // 表现成"提示一闪就没了"。
      const paint = () => { box.innerHTML = meta + Track.fixLine(Track.frameAt(vid.currentTime), vid.currentTime); };

      let raf = 0;
      const tick = () => {
        // 视频被重新渲染（innerHTML 换掉）后 rAF 循环要自己退出，
        // 否则会一直对着一个已经脱离文档的元素空转。
        if (!vid.isConnected) { raf = 0; return; }
        paint();
        raf = requestAnimationFrame(tick);
      };
      const start = () => { if (!raf) raf = requestAnimationFrame(tick); };
      const stopLoop = () => { if (raf) { cancelAnimationFrame(raf); raf = 0; } paint(); };

      vid.addEventListener('play', start);
      vid.addEventListener('playing', start);
      vid.addEventListener('pause', stopLoop);
      vid.addEventListener('ended', stopLoop);
      vid.addEventListener('timeupdate', paint);   // 暂停时拖进度条靠它
      vid.addEventListener('seeked', paint);
      paint();
    },

    async refresh() {
      const r = await Api.videoJobs();
      if (!r.ok) {
        if (!S.api.online) toast('warn', '后端未连接', '无法获取服务端任务列表');
        else toast('warn', '未获取到任务列表', '后端可能尚未实现任务查询接口');
        return;
      }
      const list = r.data.map(x => ({
        id: x.job_id || x.id,
        status: x.status,
        name: (x.params && x.params.filename) || x.name || ('任务 ' + String(x.job_id || x.id).slice(0, 8)),
        createdAt: x.created_at || x.createdAt,
        progress: x.progress || { done: 0, total: 0 },
        result: x.result || {},
        error: x.error || null,
        params: x.params || {},
      }));
      // 合并本地任务
      const seen = {};
      list.forEach(x => { seen[x.id] = 1; });
      S.jobs.forEach(x => { if (!seen[x.id]) list.push(x); });
      S.jobs = list;
      Jobs.render();
      // 分析视图的「视频任务汇总」也吃这份列表，刷新后要跟着更新，
      // 否则用户在分析页点刷新，任务数变了但汇总数字不动。
      if (S.view === 'analytics') Analytics.renderJobs();
    },

    async poll(job) {
      const tick = async () => {
        const r = await Api.videoJob(job.id, job.statusUrl);
        if (!r.ok) return;
        const d = r.data;
        job.status = d.status;
        job.progress = d.progress || job.progress;
        job.result = d.result || job.result;
        job.error = d.error || null;
        Jobs.render();
        if (job.status === 'processing' || job.status === 'queued') {
          job._timer = setTimeout(tick, 1500);
        } else if (job.status === 'completed') {
          toast('ok', '视频任务完成', (job.result && job.result.stats ? job.result.stats.total + ' 个目标' : ''));
        } else if (job.status === 'failed') {
          toast('bad', '视频任务失败', job.error || '');
        }
      };
      tick();
    },

    sel: null,
  };

  async function submitVideo(file) {
    if (!S.api.online) {
      toast('warn', '后端未连接', '视频推理需要服务端支持。当前为离线演示形态。');
      return;
    }
    toast('info', '正在提交视频', file.name + ' · ' + U.fmtBytes(file.size));
    const r = await Api.submitVideo(file, S.params);
    if (!r.ok) { toast('bad', '视频提交失败', r.error); return; }
    const job = {
      id: r.jobId || U.uid(),
      name: file.name,
      size: file.size,
      status: r.status || (r.async ? 'queued' : 'completed'),
      createdAt: Date.now(),
      progress: { done: 0, total: 0 },
      result: r.async ? {} : { video_url: r.videoUrl, stats: r.stats },
      statusUrl: r.statusUrl,
      params: Object.assign({}, S.params),
      error: null,
    };
    S.jobs.unshift(job);
    Jobs.sel = job.id;
    toast('ok', r.async ? '已提交后台任务' : '视频处理完成', r.async ? '可在「任务」视图查看进度' : '');
    if (r.async) Jobs.poll(job);
    go('jobs');
  }

  /* ======================================================== 分析视图（全系统汇总） */
  /**
   * 这个视图的定位是**汇总**：系统各处算出来的东西都收在这里。
   *
   * 四段来源不同、口径也不同，所以必须分段并各自标注来源 —— 混在一起会让人
   * 拿"本次会话的 12 个目标"去跟"训练的 mAP50"对比，那是对不上的：
   *
   * 1. 本次会话检测  —— 本机浏览器累计（`S.session`），只在内存/localStorage 里；
   * 2. 模型与训练    —— 后端只读解析训练产物（`/api/v1/metrics`，原「设置」页搬来）；
   * 3. 视频任务汇总  —— 后端任务列表（`/api/v1/jobs`，逐帧检测 + 人群密度）；
   * 4. 定位与轨迹    —— 北斗模块实时帧（`/api/v1/beidou`）+ 导入的存储卡日志。
   *
   * 任何一段拿不到数据都要显示**明确的空状态**（并说明去哪儿产生数据），
   * 而不是留一片空白让人以为坏了。
   */
  const Analytics = {
    render() {
      this.renderSession();
      this.renderJobs();
      this.renderGeo();
      // 训练指标自己触发加载：它原来挂在设置页，搬过来之后必须由本视图负责。
      Train.load();
    },

    /* ------------------------------------------------ 1. 本次会话检测 */
    renderSession() {
      const s = S.session;
      $('#an-frames').textContent = U.fmtNum(s.frames);
      $('#an-dets').textContent = U.fmtNum(s.dets);
      $('#an-dets-d').textContent = s.frames ? '平均每帧 ' + (s.dets / s.frames).toFixed(1) + ' 个' : '—';
      const avg = s.confCount ? s.sumConf / s.confCount : null;
      $('#an-avgconf').textContent = avg == null ? '—' : (avg * 100).toFixed(1) + '%';
      const lat = s.latencies.slice().sort((a, b) => a - b);
      const p95 = lat.length ? lat[Math.min(lat.length - 1, Math.floor(lat.length * 0.95))] : null;
      $('#an-p95').innerHTML = (p95 == null ? '—' : Math.round(p95)) + '<small>ms</small>';
      $('#an-p95-d').textContent = lat.length
        ? '中位 ' + Math.round(lat[Math.floor(lat.length / 2)]) + ' ms · 样本 ' + lat.length
        : '—';

      const items = CLASSES.map(c => ({ label: c.zh, value: s.classCount[c.id] || 0, color: BDS.classColor(c.id) }));
      Charts.donut($('#ch-donut'), items, { size: 168, thickness: 16, centerLabel: '目标总数' });
      $('#an-donut-total').textContent = U.fmtNum(s.dets);

      Charts.hbar($('#ch-classbar'), items.slice().sort((a, b) => b.value - a.value), { width: 620, rowH: 24 });

      Charts.sparkline($('#ch-latency'), s.latencies.slice(-80), { width: 660, height: 168, unit: '' });
      $('#an-lat-n').textContent = s.latencies.length + ' 次';

      const bins = s.confHist.map((v, i) => ({
        label: (i * 0.05).toFixed(2),
        value: v,
        color: i < 5 ? 'var(--warn)' : i < 12 ? 'var(--accent)' : 'var(--ok)',
      }));
      Charts.histogram($('#ch-confhist'), bins, { width: 560, height: 168 });

      // 表格
      const total = s.dets || 1;
      const rows = CLASSES.map(c => {
        const n = s.classCount[c.id] || 0;
        return { c: c, n: n, pct: n / total * 100 };
      }).sort((a, b) => b.n - a.n);
      $('#an-table').innerHTML = s.dets ? ('<table class="tbl"><thead><tr>' +
        '<th>类别</th><th class="mono">EN</th><th class="r">检出数</th><th class="r">占比</th>' +
        '<th>分布</th></tr></thead><tbody>' +
        rows.map(r =>
          '<tr><td><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:' +
          BDS.classColor(r.c.id) + ';margin-right:7px"></span>' + U.esc(r.c.zh) + '</td>' +
          '<td class="mono" style="color:var(--fg-4)">' + U.esc(r.c.en) + '</td>' +
          '<td class="mono r">' + U.fmtNum(r.n) + '</td>' +
          '<td class="mono r">' + r.pct.toFixed(1) + '%</td>' +
          '<td style="width:40%"><span style="display:block;height:6px;border-radius:3px;background:var(--panel-3);overflow:hidden">' +
          '<i style="display:block;height:100%;width:' + r.pct.toFixed(1) + '%;background:' + BDS.classColor(r.c.id) + '"></i></span></td>' +
          '</tr>').join('') +
        '</tbody></table>') : Analytics.emptyTable('本机还没有推理记录', '在「检测台」跑一次检测，这里就会累计。');
      const note = $('#an-table-note');
      if (note) note.textContent = s.frames ? s.frames + ' 帧 · ' + s.latencies.length + ' 次推理' : '—';

      // 数据统计（设置页）
      const ds = $('#data-stat');
      if (ds) ds.textContent = s.frames + ' 帧 · ' + U.fmtNum(s.dets) + ' 个目标 · ' + s.latencies.length + ' 次推理';
    },

    /** 空表格占位：一句话 + 去哪儿产生数据。空白区比"没有数据"更难理解。 */
    emptyTable(title, hint) {
      return '<div class="an-empty"><b>' + U.esc(title) + '</b><span>' + U.esc(hint) + '</span></div>';
    },

    /* ------------------------------------------------ 3. 视频任务汇总 */
    /**
     * 把 `S.jobs` 里已完成任务的逐帧统计**加起来**。
     *
     * 口径注意：`stats.frames` 是任务各自的帧数，累加得到的是"处理过的帧次数"
     * 而不是"多少段视频" —— 界面上按"累计处理帧"写，不写"总时长"，
     * 因为帧率可能不同（不同视频 fps 不一样），换算成时长是假的。
     */
    summarizeJobs() {
      const done = S.jobs.filter(j => j.status === 'completed' && j.result && j.result.stats);
      const acc = {
        tasks: done.length,
        running: S.jobs.filter(j => j.status === 'processing' || j.status === 'queued').length,
        failed: S.jobs.filter(j => j.status === 'failed').length,
        frames: 0, dets: 0,
        classCount: new Array(CLASSES.length).fill(0),
        peakDensity: null, peakTask: null,
        peakCount: null, peakCountTask: null,
        alertFrames: 0, calibrated: false, levels: {},
      };
      done.forEach(j => {
        const st = j.result.stats;
        acc.frames += +st.frames || 0;
        acc.dets += +st.total || 0;
        (st.class_count || []).forEach((n, i) => {
          if (i < acc.classCount.length) acc.classCount[i] += (+n || 0);
        });
        const c = st.crowd;
        if (!c || !c.frames) return;
        acc.alertFrames += +c.alert_frames || 0;
        if (c.calibrated) acc.calibrated = true;
        if (c.peak_count != null && (acc.peakCount == null || c.peak_count > acc.peakCount)) {
          acc.peakCount = c.peak_count; acc.peakCountTask = j;
        }
        // 密度只有标定过的任务才有 —— 未标定任务不参与峰值比较，
        // 否则"未标定"会被当成 0 混进最大值里，算出一个偏低的峰值。
        if (c.peak_density != null && (acc.peakDensity == null || c.peak_density > acc.peakDensity)) {
          acc.peakDensity = c.peak_density; acc.peakTask = j;
        }
        const k = (c.peak_level && c.peak_level.key) || 'unknown';
        acc.levels[k] = (acc.levels[k] || 0) + 1;
      });
      return acc;
    },

    renderJobs() {
      const a = this.summarizeJobs();
      const note = $('#an-jobs-note');
      if (note) {
        note.textContent = a.tasks + ' 完成' +
          (a.running ? ' · ' + a.running + ' 进行中' : '') +
          (a.failed ? ' · ' + a.failed + ' 失败' : '');
      }
      $('#aj-tasks').textContent = U.fmtNum(a.tasks);
      $('#aj-tasks-d').textContent = S.jobs.length ? '共 ' + S.jobs.length + ' 个任务' : '尚无任务';
      $('#aj-frames').textContent = U.fmtNum(a.frames);
      $('#aj-frames-d').textContent = a.tasks ? '跨 ' + a.tasks + ' 段视频' : '—';
      $('#aj-dets').textContent = U.fmtNum(a.dets);
      $('#aj-dets-d').textContent = a.frames ? '平均每帧 ' + (a.dets / a.frames).toFixed(1) + ' 个' : '—';

      // 峰值密度：只有标定过的任务才有。没有就如实写"未标定"，不给 0。
      const peakEl = $('#aj-peak'), peakD = $('#aj-peak-d');
      if (a.peakDensity != null) {
        peakEl.textContent = a.peakDensity.toFixed(2);
        peakEl.insertAdjacentHTML('beforeend', '<small>人/m²</small>');
        peakD.textContent = '峰值人数 ' + a.peakCount + ' · ' + (a.peakTask ? a.peakTask.name : '');
      } else if (a.peakCount != null) {
        peakEl.innerHTML = U.fmtNum(a.peakCount) + '<small>人</small>';
        peakD.textContent = '峰值人数（未标定面积，无密度）';
      } else {
        peakEl.textContent = '—';
        peakD.textContent = '无已完成任务';
      }

      const items = CLASSES.map(c => ({ label: c.zh, value: a.classCount[c.id] || 0, color: BDS.classColor(c.id) }));
      Charts.donut($('#ch-aj-donut'), items, { size: 168, thickness: 16, centerLabel: '任务目标总数' });
      $('#aj-donut-total').textContent = U.fmtNum(a.dets);

      // 各任务检出量：横向条形图，按总数排序
      const per = S.jobs
        .filter(j => j.result && j.result.stats)
        .map(j => ({
          label: (j.name || String(j.id).slice(0, 8)),
          value: +j.result.stats.total || 0,
          color: j.status === 'failed' ? 'var(--danger)' : 'var(--accent)',
        }))
        .sort((x, y) => y.value - x.value).slice(0, 12);
      if (per.length) Charts.hbar($('#ch-aj-tasks'), per, { width: 620, rowH: 24 });
      else $('#ch-aj-tasks').innerHTML = Analytics.emptyTable('没有可统计的任务', '在「检测台」提交一段视频，任务完成后统计会出现在这里。');

      // 任务明细表
      const rows = S.jobs.filter(j => j.result && j.result.stats).map(j => {
        const st = j.result.stats, c = st.crowd || {};
        const lv = c.peak_level || null;
        return {
          name: j.name || String(j.id).slice(0, 8),
          frames: +st.frames || 0,
          total: +st.total || 0,
          avg: st.avg_per_frame,
          peakCount: c.peak_count,
          peakDensity: c.calibrated ? c.peak_density : null,
          lv: lv,
        };
      });
      $('#aj-table').innerHTML = rows.length ? ('<table class="tbl"><thead><tr>' +
        '<th>文件</th><th class="r">帧</th><th class="r">检出</th><th class="r">平均每帧</th>' +
        '<th class="r">峰值人数</th><th class="r">峰值密度</th><th>峰值分级</th></tr></thead><tbody>' +
        rows.map(r =>
          '<tr><td>' + U.esc(r.name) + '</td>' +
          '<td class="mono r">' + U.fmtNum(r.frames) + '</td>' +
          '<td class="mono r">' + U.fmtNum(r.total) + '</td>' +
          '<td class="mono r">' + (r.avg == null ? '—' : r.avg) + '</td>' +
          '<td class="mono r">' + (r.peakCount == null ? '—' : r.peakCount) + '</td>' +
          '<td class="mono r">' + (r.peakDensity == null ? '<span style="color:var(--fg-4)">未标定</span>'
            : r.peakDensity.toFixed(2) + ' 人/m²') + '</td>' +
          '<td>' + (r.lv ? '<span class="chip ' + (CROWD_COLOR[r.lv.key] || '') +
            (r.lv.alert ? ' live' : '') + '"><span class="dot"></span>' + U.esc(r.lv.zh) + '</span>'
            : '<span style="color:var(--fg-4)">—</span>') + '</td></tr>').join('') +
        '</tbody></table>') :
        Analytics.emptyTable('没有已完成的任务', '在「检测台」提交视频后会出现在「任务」视图，完成后这里的统计自动汇总。');
    },

    /* ------------------------------------------------ 4. 定位与轨迹 */
    renderGeo() {
      const b = S.beidou || {};
      const c = fixChip(b);
      const chip = $('#an-geo-chip');
      if (chip) {
        chip.className = 'chip ' + c.cls + (b.usable && b.source === 'serial' ? ' live' : '');
        chip.innerHTML = '<span class="dot"></span>' + U.esc(c.text);
      }

      const gsdTxt = (b.gsd_m_per_px > 0)
        ? Number(b.gsd_m_per_px).toFixed(4) + ' m/px' + (b.gsd_calibrated ? ' · 已标定' : ' · 估算')
        : '未标定';
      const rows = [
        ['数据源', b.source_detail || b.source || '—'],
        ['定位质量', b.fix_quality_zh || '—'],
        ['卫星 已用 / 可见',
          (b.satellites_used == null ? '—' : b.satellites_used) + ' / ' +
          (b.satellites_visible == null ? '—' : b.satellites_visible)],
        ['HDOP', nz(b.hdop, 2)],
        ['PDOP', nz(b.pdop, 2)],
        ['纬度', b.lat == null ? '—' : Number(b.lat).toFixed(6)],
        ['经度', b.lon == null ? '—' : Number(b.lon).toFixed(6)],
        ['海拔', nz(b.alt, 1, ' m')],
        ['解算时刻', b.utc || '—'],
        ['数据龄期', b.age_s == null ? '—' : nz(b.age_s, 1, ' s')],
        ['地面采样', gsdTxt],
      ];
      const tiles = $('#an-geo-tiles');
      if (tiles) {
        tiles.className = 'geo-meta';
        tiles.innerHTML = rows.map(r =>
          '<div class="meta-i"><span class="k">' + U.esc(r[0]) + '</span>' +
          '<span class="v">' + U.esc(String(r[1])) + '</span></div>').join('');
      }
      const note = $('#an-geo-note');
      if (note) note.textContent = b.usable ? '实时定位可用' : (b.message || '未接入定位源');

      // ---- 存储卡日志（离线轨迹）摘要 ----
      const t = S.track || {};
      const d = t.diagnose, s = t.summary;
      $('#an-trk-name').textContent = t.loaded ? (t.name || '已导入') : '未导入';
      const grid = $('#an-trk-grid');
      const hints = $('#an-trk-hints');
      if (!d || !s) {
        grid.innerHTML = '<div class="an-empty"><b>还没有导入存储卡日志</b>' +
          '<span>到「北斗」视图把 MCU 写卡的日志拖进去，这里会汇总解析结果与轨迹。</span></div>';
        hints.innerHTML = '';
      } else {
        const rows2 = [
          ['识别到的格式', d.stamp_kind_zh || '—'],
          ['定位点', d.count != null ? U.fmtNum(d.count) : '—'],
          ['时长', nz(d.duration_s, 1, ' s')],
          ['历元间隔', nz(d.epoch_interval_s, 2, ' s')],
          ['时间范围', Track.timeRange(d.first_utc, d.last_utc)],
          ['是否绝对时间', d.absolute ? '是（可与视频对齐）' : '否（只能相对回放）'],
          ['对齐状态', s.aligned ? '已对齐 · 平移 ' + nz(s.shift_s, 1, ' s') : '未对齐（按起点回放）'],
          ['总行数', d.total_lines != null ? U.fmtNum(d.total_lines) : '—'],
          ['注释 / 空行', d.comment_lines != null ? U.fmtNum(d.comment_lines) : '—'],
          ['未识别行', d.skipped_lines != null ? U.fmtNum(d.skipped_lines) : '—'],
        ];
        if (d.clock_offset_s != null && Math.abs(d.clock_offset_s) >= 1) {
          rows2.push(['MCU 时钟偏移', nz(d.clock_offset_s, 1, ' s')]);
        }
        grid.className = 'geo-meta';
        grid.innerHTML = rows2.map(r =>
          '<div class="meta-i' + (r[0] === '时间范围' ? ' wide' : '') + '">' +
          '<span class="k">' + U.esc(r[0]) + '</span>' +
          '<span class="v">' + U.esc(String(r[1])) + '</span></div>').join('');
        hints.innerHTML = (d.hints && d.hints.length)
          ? '<div class="rcpt-notes" style="border-top:0;padding-left:0;padding-right:0"><ul>' +
            d.hints.map(h => '<li>' + U.esc(h) + '</li>').join('') + '</ul></div>'
          : '';
      }

      // ---- 轨迹走向 ----
      const q = Geo.trackEnu();
      Beidou.scatter($('#an-trk-scatter'), [], {
        width: 620, height: 280, track: q.track, marker: q.marker,
        emptyText: t.loaded ? '这条轨迹没有可用的经纬度点' : '导入存储卡日志后，这里会画出轨迹走向',
      });
      const tag = $('#an-trk-scatter-tag');
      if (tag) {
        tag.textContent = q.track.length
          ? '东 / 北 · 米 · 轨迹 ' + q.track.length + ' 点'
          : '—';
      }
    },
  };

  /* ======================================================== 训练成果（真实训练产物） */
  /**
   * 数据全部来自后端 /api/v1/metrics —— 后端只读解析已加载权重所属 run 的
   * results.csv / args.yaml / 消融对比表，前端不做任何硬编码指标。
   */
  const Train = {
    data: null,
    loading: false,
    error: null,

    /** 拉取一次并缓存；force=true 强制重取。失败只提示，不打断页面。 */
    async load(force) {
      if (!$('#train-kpis') || this.loading) return;
      if (!force && this.data) { this.render(); return; }

      if (!S.api.online) {
        this.data = null;
        this.error = '未连接后端：训练指标由后端读取训练产物目录后提供。';
        this.render();
        return;
      }
      this.loading = true;
      const rt = $('#train-run');
      if (rt) rt.textContent = '读取中…';
      const r = await Api.metrics();
      this.loading = false;
      if (r.ok) { this.data = r.data; this.error = null; }
      else { this.data = null; this.error = '指标接口不可用（后端需提供 /api/v1/metrics）。'; }
      this.render();
    },

    render() {
      const kpis = $('#train-kpis');
      if (!kpis) return;
      const d = this.data;
      const pct = v => (v == null || isNaN(v) ? '—' : (v * 100).toFixed(1) + '%');
      const tiles = kpis.querySelectorAll('.kpi');
      const setTile = (i, v, dsc) => {
        if (!tiles[i]) return;
        tiles[i].querySelector('.v').innerHTML = v;
        tiles[i].querySelector('.d').textContent = dsc || '';
      };

      // ---- 不可用 / 空状态：清空所有区块，避免残留上一次的数据 ----
      if (!d || !d.available) {
        for (let i = 0; i < tiles.length; i++) setTile(i, '—', '—');
        $('#train-curve').innerHTML = '';
        $('#train-curve-sub').textContent = '—';
        $('#train-cmp').innerHTML = '';
        $('#train-cmp-sub').textContent = '—';
        $('#train-art').innerHTML = '';
        $('#train-art-row').hidden = true;
        const bn0 = $('#train-best-note');
        if (bn0) { bn0.hidden = true; bn0.innerHTML = ''; }
        // 段头那句"来源：…"是**静态说明**，不能被动态状态覆盖 ——
        // 一覆盖，用户就看不出这一块的数据本该从哪儿来了，
        // 而且"没有数据"和"数据来源变了"会长得一模一样。
        // 动态状态统一放段头右侧的 tag。
        const rt0 = $('#train-run');
        if (rt0) rt0.textContent = '不可用';
        $('#train-note').textContent = (d && d.reason) || this.error || '尚未连接后端。';
        return;
      }

      const b = d.best || {};
      setTile(0, pct(b.mAP50), b.epoch != null ? 'best @ epoch ' + b.epoch : '');
      setTile(1, pct(b.mAP50_95), 'P ' + pct(b.precision) + ' · R ' + pct(b.recall));
      setTile(2, d.params == null ? '—' : U.fmtNum(d.params), d.layers ? d.layers + ' 个模块' : '');
      setTile(3, U.fmtNum(d.epochs), d.args && d.args.epochs ? '计划 ' + d.args.epochs : '');
      const runTag = $('#train-run');
      if (runTag) runTag.textContent = (d.run || '—') + ' · ' + d.epochs + ' epoch';

      // ---- 两个「最优」不是同一轮，必须讲清楚 ----
      // KPI 取「mAP50 峰值」那一轮；best.pt 是按 fitness（0.1*mAP50 + 0.9*mAP50-95）
      // 选的另一轮。不说明的话，KPI 的 44.6% 会和 PR 曲线上的 44.2% 看起来自相矛盾
      // —— 用户会以为图是从别处拼凑来的（真实发生过）。
      const bf = d.best_fitness;
      const bn = $('#train-best-note');
      if (bn) {
        if (bf && bf.epoch != null && b.epoch != null && bf.epoch !== b.epoch) {
          bn.hidden = false;
          bn.innerHTML = '<b>关于两个「最优」</b>（口径不同，都对）：<ul>' +
            '<li>上方 KPI 与收敛曲线取 <b>mAP50 峰值</b>那一轮：epoch ' + b.epoch +
            '，mAP50 ' + pct(b.mAP50) + '。</li>' +
            '<li>权重文件 <code>best.pt</code> 存的是 <b>fitness 最优</b>那一轮：epoch ' + bf.epoch +
            '，mAP50 ' + pct(bf.mAP50) + '（fitness ' + (bf.fitness == null ? '—' : bf.fitness.toFixed(5)) + '）。' +
            'fitness = 0.1·mAP50 + 0.9·mAP50-95，所以它会挑 mAP50-95 更高、而 mAP50 略低的那一轮。</li>' +
            '<li>下方 PR / F1 / 混淆矩阵是用 <code>best.pt</code> 复评生成的，因此图上写的是 ' +
            pct(bf.mAP50) + ' 而不是 ' + pct(b.mAP50) + ' —— 两者不是同一次测量，不是数据错误。</li>' +
            '</ul>';
        } else if (bf) {
          bn.hidden = false;
          bn.innerHTML = 'mAP50 峰值与 fitness 最优落在同一轮（epoch ' + b.epoch +
            '），所以 KPI、曲线与 PR 曲线上的读数一致。';
        } else {
          bn.hidden = true;
          bn.innerHTML = '';
        }
      }

      // ---- 收敛曲线 ----
      const cur = d.curves || {};
      Charts.curve($('#train-curve'), [
        { name: 'mAP50', values: cur.mAP50 || [], color: 'var(--accent)' },
        { name: 'mAP50-95', values: cur.mAP50_95 || [], color: 'var(--ok)' },
        { name: 'precision', values: cur.precision || [], color: 'var(--warn)', dash: '4 3', width: 1.4, opacity: 0.9 },
        { name: 'recall', values: cur.recall || [], color: 'var(--danger)', dash: '4 3', width: 1.4, opacity: 0.9 },
      ], {
        width: 700, height: 236, xStart: 1, xTicks: 6,
        fmt: v => v.toFixed(2), yMin: 0, yFloor0: true,
        bestPrefix: 'best mAP50 ', empty: '该 run 无 results.csv',
      });
      $('#train-curve-sub').textContent = '横轴 epoch 1–' + d.epochs + '；圆点标注 mAP50 峰值轮次。';

      // ---- 消融对比 ----
      const cmp = d.comparison;
      const cmpHost = $('#train-cmp');
      if (cmp && cmp.items && cmp.items.length) {
        const myTag = (d.run || '').replace('visdrone_', '');
        cmpHost.innerHTML = '<table class="tbl"><thead><tr>' +
          '<th>变体</th><th class="r">mAP50</th><th class="r">mAP50-95</th>' +
          '<th class="r">P</th><th class="r">R</th><th class="r">参数量</th></tr></thead><tbody>' +
          cmp.items.map(it => {
            const on = myTag && it.name.toLowerCase().indexOf(myTag.toLowerCase()) >= 0;
            return '<tr' + (on ? ' style="background:var(--accent-soft)"' : '') + '>' +
              '<td>' + U.esc(it.name) + (on ? ' <span class="tag">当前</span>' : '') + '</td>' +
              '<td class="mono r">' + (it.mAP50 * 100).toFixed(2) + '</td>' +
              '<td class="mono r">' + (it.mAP50_95 * 100).toFixed(2) + '</td>' +
              '<td class="mono r">' + (it.precision * 100).toFixed(2) + '</td>' +
              '<td class="mono r">' + (it.recall * 100).toFixed(2) + '</td>' +
              '<td class="mono r">' + U.fmtNum(it.params) + '</td></tr>';
          }).join('') +
          '</tbody></table>' +
          (cmp.notes && cmp.notes.length
            ? '<div class="train-note" style="margin-top:10px"><ul>' +
              cmp.notes.map(n => '<li' + (/⚠|不严谨/.test(n) ? ' class="warn"' : '') + '>' +
                U.esc(n) + '</li>').join('') + '</ul></div>'
            : '');
        $('#train-cmp-sub').textContent = '来源 ' + cmp.name + ' · ' + cmp.items.length + ' 个变体（数值为 %）';
      } else {
        cmpHost.innerHTML = '<div class="train-note">run 目录上一级未找到对比汇总文件。</div>';
        $('#train-cmp-sub').textContent = '—';
      }

      // ---- 训练配图（后端直出，不走 base64）----
      // 优先用项目内重绘图目录（MATLAB 重绘曲线 + 当前权重重跑验证的 PR/混淆矩阵），
      // 为空才回落到 run 目录里的训练原图 —— 后者与 KPI 不同源，会出 44.6%/44.2% 的错觉。
      const art = d.artifacts || [];
      const fromFigs = d.artifacts_source === 'figures';
      $('#train-art-row').hidden = !art.length;
      $('#train-art').innerHTML = art.map(a =>
        '<figure data-art="' + U.esc(a.name) + '" data-cap="' + U.esc(a.caption || a.name) + '" title="点击放大">' +
        '<img loading="lazy" src="' + U.esc(Api.metricsArtifact(a.name)) + '" alt="' + U.esc(a.caption || a.name) + '">' +
        '<figcaption>' + U.esc(a.caption || a.name) + '</figcaption></figure>').join('');
      U.$$('#train-art figure').forEach(f => { f.onclick = () => openLightbox(f.dataset.art, f.dataset.cap); });
      const asub = $('#train-art-sub');
      if (asub) {
        asub.textContent = fromFigs
          ? '项目重绘 · ' + art.length + ' 张 · 点击放大'
          : '训练原图（回退）· ' + art.length + ' 张 · 点击放大';
      }

      // ---- 来源备注 ----
      const a = d.args || {};
      const lines = [
        'run 目录 <code>' + U.esc(d.run_dir || '—') + '</code>',
        '权重 <code>' + U.esc(d.weights || '—') + '</code>',
      ];
      if (a.model) {
        lines.push('架构 <code>' + U.esc(a.model) + '</code> · imgsz ' + U.esc(a.imgsz) +
          ' · batch ' + U.esc(a.batch) + ' · ' + U.esc(a.optimizer) + ' · lr0 ' + U.esc(a.lr0));
      }
      $('#train-note').innerHTML = lines.join('<br>');
    },
  };
  BDS.Train = Train;   // 暴露给 dev 探针与调试用
  BDS.Analytics = Analytics;
  BDS.runInference = runInference;

  /** 配图放大：复用 modal，图片走后端直出地址 */
  function openLightbox(name, caption) {
    if (!name || !Train.data) return;
    const url = Api.metricsArtifact(name);
    $('#lb-title').textContent = caption || name;
    $('#lb-img').src = url;
    $('#lb-open').href = url;
    $('#lb-cap').textContent = (Train.data.run || '') + ' · ' + name + ' · 后端直出原图';
    openModal('modal-lightbox');
  }

  /* ======================================================== 北斗视图 */
  /** 可空数值格式化：null / undefined / 非数 → "—"。绝不退化成 0。 */
  function nz(v, digits, suffix) {
    if (v == null || v === '') return '—';
    const n = Number(v);
    if (!isFinite(n)) return '—';
    return n.toFixed(digits == null ? 1 : digits) + (suffix || '');
  }

  /**
   * 定位状态 → chip 文案与修饰类。
   * 修饰类只能用真实存在的 CSS 类（ok / warn / bad / accent / geo / live）：
   * 写错不报错，但 getComputedStyle 拿到的是灰色，用户根本看不出严重程度。
   */
  function fixChip(b) {
    if (!b || b.source === 'unavailable') return { cls: 'bad', text: '未接入定位源' };
    if (!b.usable) return { cls: 'bad', text: b.fix_quality_zh || '无定位' };
    if (b.stale) return { cls: 'warn', text: '定位过期' };
    if (b.source === 'mock') return { cls: 'geo', text: '模拟定位' };
    if (b.source === 'file') return { cls: 'ok', text: '日志回放定位' };
    return { cls: 'ok', text: '实时定位' };
  }

  const Geo = {
    renderTiles() {
      const b = S.beidou || {};
      const ok = !!b.usable && b.lat != null && b.lon != null;

      $('#geo-lat').textContent = ok ? Number(b.lat).toFixed(6) : '—';
      $('#geo-lon').textContent = ok ? Number(b.lon).toFixed(6) : '—';
      $('#geo-alt').innerHTML = (ok && b.alt != null)
        ? Number(b.alt).toFixed(1) + '<small>m</small>' : '—<small>m</small>';

      // 「已用 / 可见」是两个不同口径，各自都可能缺 —— 缺哪个就在哪个位置写 —
      const used = b.satellites_used, vis = b.satellites_visible;
      $('#geo-sats').textContent = (used == null && vis == null)
        ? '—' : (used == null ? '—' : used) + ' / ' + (vis == null ? '—' : vis);
      const cons = [];
      (b.satellites_detail || []).forEach(s => {
        const k = Beidou.TALKER[String(s.talker || '').toUpperCase()] || 'GNSS';
        if (cons.indexOf(k) < 0) cons.push(k);
      });
      $('#geo-sats-d').textContent = cons.length ? cons.join(' + ') : '颗';

      const c = fixChip(b);
      const chip = $('#geo-real-chip');
      chip.className = 'chip ' + c.cls + (b.usable && b.source === 'serial' ? ' live' : '');
      chip.innerHTML = '<span class="dot"></span>' + U.esc(c.text);

      // 定位质量与健康度明细
      const gsdTxt = (b.gsd_m_per_px > 0)
        ? Number(b.gsd_m_per_px).toFixed(4) + ' m/px' + (b.gsd_calibrated ? ' · 已标定' : ' · 估算')
        : '未标定';
      const rows = [
        ['定位质量', b.fix_quality_zh || '—'],
        ['定位模式', b.mode_zh || '—'],
        ['HDOP', nz(b.hdop, 2)],
        ['PDOP', nz(b.pdop, 2)],
        ['解算时刻', b.utc || '—'],
        ['速度', nz(b.speed_kmh, 1, ' km/h')],
        ['航向', nz(b.course, 1, '°')],
        ['数据源', b.source_detail || b.source || '—'],
        ['数据龄期', b.age_s == null ? '—' : nz(b.age_s, 1, ' s')],
        ['地面采样', gsdTxt],
      ];
      $('#geo-meta').innerHTML = rows.map(r =>
        '<div class="meta-i"><span class="k">' + U.esc(r[0]) + '</span>' +
        '<span class="v">' + U.esc(String(r[1])) + '</span></div>').join('');

      // 提示语：后端 message 是排查串口/接线问题的第一手线索，原样透出
      const msgs = [];
      if (b.message) msgs.push(b.message);
      if (b.checksum_errors > 0) msgs.push('已丢弃 ' + b.checksum_errors + ' 条校验失败的语句。');
      if (ok && b.estimated) {
        msgs.push('目标坐标按估算地面采样距离（' + gsdTxt + '）推算，属示意值；' +
          '按实际飞行高度标定 GEO_GSD_M_PER_PX 后才算测量值。');
      }
      const msgEl = $('#geo-msg');
      msgEl.textContent = msgs.join(' ');
      msgEl.style.display = msgs.length ? '' : 'none';

      Geo.renderBds(b);
    },

    /**
     * 北斗特色服务三块：精度档位 / 时间基准 / 星座构成。
     *
     * 这三块的存在意义是回答"**除了定位，北斗还提供了什么**"：
     *   - 档位：同样是"有定位"，单点解与 RTK 固定解差了三个数量级
     *   - 时间基准：UTC 不是拿来显示的，是拿来**核对**的
     *   - 星座构成：说明北斗在这一帧里出了多少力（多星座兼容是北斗的特性）
     *
     * 判不出来一律 "—" + 原因。**绝不用 0 或默认档位填坑** ——
     * 一个显示"单点定位"的默认值，会让用户以为精度已经评估过了。
     */
    renderBds(b) {
      const host = $('#geo-bds-grid');
      const chip = $('#geo-tier-chip');
      const hint = $('#geo-bds-hint');
      if (!host) return;
      b = b || {};

      const t = b.accuracy_tier || null;
      const tb = b.timebase || null;
      const cp = b.composition || null;
      const cards = [];
      const hints = [];

      /* ---------- 1. 精度档位 ---------- */
      if (t) {
        // level_m 是**北斗该服务的系统能力量级**，不是本机实测值 —— 必须写明白
        const lv = (t.level_m != null)
          ? '该档位系统能力：水平约 ' + t.level_m + ' m 量级'
          : '该档位无公开量级指标';
        cards.push(
          '<div class="bds-card">' +
            '<div class="bds-card-hd">精度档位' +
              '<div class="sp"></div>' +
              '<span class="bds-tier t-' + U.esc(t.key) + '">' + U.esc(t.zh) + '</span>' +
            '</div>' +
            '<div class="bds-card-v">' + U.esc(t.zh) + '</div>' +
            '<div class="bds-card-d">' + U.esc(lv) +
              '<br>系统能力指标，非本机实测精度。' +
              (t.detail ? '<br>' + U.esc(t.detail) : '') +
            '</div>' +
          '</div>');
        if (chip) {
          chip.className = 'chip ' + (t.key === 'fixed' ? 'ok' : 'geo');
          chip.textContent = t.zh;
        }
      } else {
        cards.push(
          '<div class="bds-card na">' +
            '<div class="bds-card-hd">精度档位' +
              '<div class="sp"></div><span class="bds-tier">未知</span>' +
            '</div>' +
            '<div class="bds-card-v na">—</div>' +
            '<div class="bds-card-d">' + U.esc(b.accuracy_message || '暂无可判定的定位质量字段。') + '</div>' +
          '</div>');
        if (chip) { chip.className = 'chip'; chip.textContent = '档位未知'; }
      }

      /* ---------- 2. 时间基准（授时） ---------- */
      if (tb && tb.delta_s != null) {
        // delta = 北斗 − 本机。正数 = 本机慢了。符号含义写出来，不靠颜色单独表意。
        const d = Number(tb.delta_s);
        const cls = Math.abs(d) < 1 ? 'ok' : (Math.abs(d) < 10 ? 'warn' : 'bad');
        const sign = d >= 0 ? '' : '−';
        const word = d >= 0 ? '本机慢' : '本机快';
        cards.push(
          '<div class="bds-card">' +
            '<div class="bds-card-hd">时间基准（授时）<div class="sp"></div>' +
              '<span class="bds-tier t-ppp">北斗 UTC</span></div>' +
            '<div class="bds-card-v">' + U.esc(tb.utc || '—') + '</div>' +
            '<div class="bds-delta ' + cls + '">与系统时钟偏差 ' +
              '<b>' + sign + Math.abs(d).toFixed(3) + ' s</b>（' + word + '）</div>' +
            '<div class="bds-card-d">' + U.esc(tb.source_zh || '') + '</div>' +
          '</div>');
      } else {
        cards.push(
          '<div class="bds-card na">' +
            '<div class="bds-card-hd">时间基准（授时）<div class="sp"></div>' +
              '<span class="bds-tier">未建立</span></div>' +
            '<div class="bds-card-v na">—</div>' +
            '<div class="bds-card-d">' +
              U.esc((tb && tb.message) || '尚未建立北斗时间基准。') + '</div>' +
          '</div>');
        // 卡片里已经写清了原因，这里**不再重复**。
        // 只有在"时刻拿到了、但不能当基准用"这种**信息量更大**的情况下才补一句，
        // 因为此时用户会问"我明明看到有 UTC，为什么不算"。
        if (tb && tb.utc && !tb.message) {
          hints.push('已收到北斗 UTC（' + tb.utc + '），但还不能作为时间基准。');
        }
      }

      /* ---------- 3. 星座构成 ---------- */
      if (cp && cp.available && cp.visible_total) {
        const rows = [];
        const byc = cp.visible_by_constellation || {};
        Object.keys(byc).forEach(k => {
          const n = byc[k];
          const pct = Math.round(100 * n / cp.visible_total);
          // 北斗用主题紫单独标出 —— 这是"北斗主题"最直接的视觉落点
          const isBds = (k === '北斗');
          rows.push(
            '<div class="bds-cons-row' + (isBds ? ' bds' : '') + '">' +
              '<span class="n">' + U.esc(k) + '</span>' +
              '<span class="bar"><i style="width:' + pct + '%"></i></span>' +
              '<span class="c">' + n + '</span>' +
            '</div>');
        });
        let shareTxt;
        if (cp.beidou_visible == null) {
          shareTxt = '北斗占比不可判定';
        } else {
          shareTxt = '北斗 ' + cp.beidou_visible + ' 颗 · 占可见 ' +
            Math.round((cp.beidou_share || 0) * 100) + '%';
        }
        cards.push(
          '<div class="bds-card">' +
            '<div class="bds-card-hd">星座构成<div class="sp"></div>' +
              '<span class="bds-tier t-sbas">可见 ' + cp.visible_total + '</span></div>' +
            '<div class="bds-cons">' + rows.join('') + '</div>' +
            '<div class="bds-card-d">' + U.esc(shareTxt) +
              '<br>参与解算共 ' + (cp.used_total == null ? '—' : cp.used_total) +
              ' 颗（合并语句无法按星座拆分）。</div>' +
          '</div>');
        if (cp.message) hints.push(cp.message);
      } else {
        cards.push(
          '<div class="bds-card na">' +
            '<div class="bds-card-hd">星座构成<div class="sp"></div>' +
              '<span class="bds-tier">无数据</span></div>' +
            '<div class="bds-card-v na">—</div>' +
            '<div class="bds-card-d">' +
              U.esc((cp && cp.message) || '未收到卫星语句，无法统计星座构成。') + '</div>' +
          '</div>');
      }

      host.innerHTML = cards.join('');
      hint.textContent = hints.join(' ');
      hint.style.display = hints.length ? '' : 'none';
    },

    /**
     * 星空图。有 GSV 实测就用实测，没有才退回生成星座。
     *
     * 退回生成星座**只在明确是模拟源时**发生：给一台没插模块的机器画一张
     * 有 12 颗卫星的星空图，哪怕标签写着"示意"，图本身也比标签更有说服力。
     * 真实源（serial / file）没收到 GSV 就画空图。
     *
     * 导入了日志时**优先跟时间轴走** —— 拖动时看到的是"当时"的星空，
     * 而不是此刻的实时读数，这两者混起来会让人误判信号质量。
     */
    renderSky() {
      const b = S.beidou || {};
      const tf = (S.track && S.track.fix) || null;
      const useTrack = !!(tf && tf.satellites && tf.satellites.length);
      const detail = useTrack ? tf.satellites : b.satellites_detail;
      const source = useTrack ? 'track' : b.source;
      const used = useTrack ? tf.satellites_used : b.satellites_used;

      const tag = $('#sky-tag');
      const real = Beidou.fromDetail(detail);

      if (real.length) {
        Beidou.skyplot($('#skyplot'), real, { size: 300 });
        if (tag) {
          tag.textContent = (useTrack ? '日志时刻 · ' : 'GSV 实测 · ') + real.length + ' 颗';
        }
        return;
      }
      if (source === 'mock') {
        const sats = Beidou.constellation(used || 12, 20260913 + (used || 0));
        Beidou.skyplot($('#skyplot'), sats, { size: 300 });
        if (tag) tag.textContent = '示意（模拟源，无 GSV 实测）';
        return;
      }
      Beidou.skyplot($('#skyplot'), [], { size: 300, empty: '未收到 GSV 卫星数据' });
      if (tag) tag.textContent = '无数据';
    },

    /**
     * 轨迹折线 + 当前位置标记的 ENU 坐标。基准点取**轨迹起点**，
     * 于是整条路线与"当前时刻在哪"落在同一套坐标里。
     *
     * 「北斗」视图与「分析」视图共用这一段 —— 两处各算一遍迟早会算出不同的
     * 比例尺（`geo_map` / `enuFromPixel` 上已经踩过一次：后端 ≈2.94 m/px、
     * 前端写死 0.021 m/px，差 140 倍，两个表并列摆着自相矛盾）。
     */
    trackEnu() {
      const tf = (S.track && S.track.fix) || null;
      const poly = (S.track && S.track.polyline) || [];
      if (!tf || tf.lat == null || tf.lon == null || !poly.length) {
        return { track: [], marker: null, refLat: null, refLon: null };
      }
      const refLat = poly[0].lat, refLon = poly[0].lon;
      const track = poly.map(p => {
        const q = BDS.enuFromGeo(p.lat, p.lon, refLat, refLon);
        return { e: q.e, n: q.n };
      }).filter(q => q.e != null);
      const m = BDS.enuFromGeo(tf.lat, tf.lon, refLat, refLon);
      const marker = (m.e == null) ? null
        : { e: m.e, n: m.n, label: '视频 ' + (+S.track.t || 0).toFixed(1) + ' s 的位置' };
      return { track, marker, refLat, refLon };
    },

    /**
     * ENU 散布图。基准点分两种情形，但换算只有一套常数：
     *
     * * 无日志：基准 = 当前定位帧（≈图像中心），目标点按像素偏移换算；
     * * 有日志：基准 = **轨迹起点**，于是整条路线、当前时刻的位置、
     *   以及此刻的目标点能落在同一张图上 —— 目标点 = 像素偏移 + 当前时刻相对起点的偏移。
     */
    renderScatter() {
      const f = activeFrame();
      const dets = (f && f.dets) ? filteredDets() : [];
      const q = Geo.trackEnu();
      const track = q.track, marker = q.marker;

      // 未标定 GSD 时 enuFromPixel 返回 null，这些点**直接丢弃**，
      // 否则会全部堆到原点，画出一张"看起来有数据"的假图。
      const pts = [];
      if (f && f.width) {
        dets.slice(0, 220).forEach(d => {
          const cx = (d.x1 + d.x2) / 2, cy = (d.y1 + d.y2) / 2;
          const enu = BDS.enuFromPixel(cx, cy, f.width, f.height);
          if (enu.e == null || enu.n == null) return;
          const c = CLASSES[d.cls] || { zh: '?' };
          pts.push({
            e: +(enu.e + (marker ? marker.e : 0)).toFixed(2),
            n: +(enu.n + (marker ? marker.n : 0)).toFixed(2),
            color: BDS.classColor(d.cls),
            label: c.zh + ' · ' + (d.conf * 100).toFixed(0) + '%',
          });
        });
      }

      Beidou.scatter($('#scatter'), pts, {
        width: 520, height: 260, track: track, marker: marker,
        emptyText: S.track && S.track.loaded ? '该时刻没有目标，也没有轨迹点' : '暂无目标定位数据',
      });
      const tag = $('#scatter-tag');
      if (tag) {
        tag.textContent = marker
          ? '东 / 北 · 米 · 轨迹 ' + track.length + ' 点 · 目标 ' + pts.length
          : '东 / 北 · 米 · ' + pts.length + ' 点';
      }
      const cnt = $('#geo-count');
      if (cnt) cnt.textContent = dets.length;
    },

    render() {
      Geo.renderTiles();
      Geo.renderSky();
      Geo.renderScatter();
      const f = activeFrame();
      const dets = (f && f.dets) ? filteredDets() : [];

      const host = $('#geo-table');
      if (!dets.length) {
        host.innerHTML = '<div class="empty"><div class="empty-ic"><svg><use href="#i-crosshair"/></svg></div>' +
          '<h4>暂无目标定位</h4><p>在「检测台」完成一次推理后，目标的地理坐标估算会列在这里。</p>' +
          '<button class="btn btn-sm" id="btn-geo-go"><svg><use href="#i-scan"/></svg>前往检测台</button></div>';
        const gb = $('#btn-geo-go');
        if (gb) gb.onclick = () => go('detect');
        return;
      }
      host.innerHTML = '<table class="tbl"><thead><tr><th>#</th><th>类别</th><th class="r">纬度 LAT</th>' +
        '<th class="r">经度 LON</th><th class="r">东 / 北 (m)</th></tr></thead><tbody>' +
        dets.slice(0, 200).map((d, i) => {
          const cx = (d.x1 + d.x2) / 2, cy = (d.y1 + d.y2) / 2;
          const g = (d.lat != null && d.lon != null)
            ? { lat: d.lat, lon: d.lon }
            : BDS.geoFromPixel(cx, cy, f.width, f.height);
          const enu = BDS.enuFromPixel(cx, cy, f.width, f.height);
          const c = CLASSES[d.cls] || { zh: '?' };
          return '<tr><td class="mono" style="color:var(--fg-4)">' + (i + 1) + '</td>' +
            '<td><span style="display:inline-block;width:7px;height:7px;border-radius:2px;background:' +
            BDS.classColor(d.cls) + ';margin-right:6px"></span>' + U.esc(c.zh) + '</td>' +
            '<td class="mono r">' + (g ? Number(g.lat).toFixed(6) : '—') + '</td>' +
            '<td class="mono r">' + (g ? Number(g.lon).toFixed(6) : '—') + '</td>' +
            '<td class="mono r">' + (enu.e == null ? '—' : enu.e.toFixed(1) + ' / ' + enu.n.toFixed(1)) + '</td></tr>';
        }).join('') +
        '</tbody></table>' +
        (dets.length > 200 ? '<div class="fld-hint" style="padding:10px 12px">仅列出前 200 条。</div>' : '');
    },
  };

  BDS.Geo = Geo;                 // 暴露给 dev 探针与调试用
  BDS.applyBeidou = applyBeidou;

  /* ======================================================== 存储卡日志回放 */
  /**
   * 离线路线：导入 MCU 写的日志 → 拖时间轴看任意时刻的定位。
   *
   * 与「实时定位」是两件事，所以各占一块、各自标注来源，不混成一套读数 ——
   * 混起来用户就分不清"现在在哪"和"当时在哪"。
   */
  const Track = {
    timer: null,
    _gen: 0,      // 回放代际号。**必须显式初始化**：`++undefined` 是 NaN，
                  // 而 `NaN !== NaN` 恒真，第一帧就会自己退出，表现为"按播放没反应"。
    _seq: 0,

    /** 拉一次服务端状态（可能是上次已经导入过的） */
    async load() {
      const r = await Api.trackState();
      if (!r.ok) return;
      this._applyState(r.data || {});
      if (S.track.loaded) this.seek(0, true);
      this.render();
    },

    /** 把服务端状态写进本地 Store（导入与刷新走同一条路径，避免两处漏字段） */
    _applyState(d) {
      S.track.loaded = !!d.loaded;
      S.track.name = d.name || null;
      S.track.error = d.error || null;
      S.track.summary = d.summary || null;
      S.track.diagnose = d.diagnose || null;
      S.track.polyline = d.polyline || [];
      S.track.scrub = d.scrub || [];
      S.track.scrubStride = +d.scrub_stride || 1;
      S.track.scrubSats = d.scrub_sats || [];
      S.track.t = 0;
      S.track.fix = null;
    },

    async importFile(file) {
      if (!file) return;
      this.stop();
      S.track.loading = true;
      this.render();
      const r = await Api.trackImport(file);
      S.track.loading = false;
      if (!r.ok) {
        // 网络/体积类错误也要走回执区：只弹个 toast，用户回头就找不到原因了
        S.track.diagnose = { ok: false, count: 0, hints: [r.error || '导入失败'], sample_lines: [], skipped_samples: [] };
        S.track.summary = null;
        S.track.loaded = false;
        S.track.scrub = [];
        S.track.scrubSats = [];
        this.render();
        toast('bad', '导入失败', r.error || '未知错误');
        return;
      }
      // 导入响应只带回执，序列要从 /track 再取一次 —— 那一份才是带 scrub 的完整状态。
      const st = await Api.trackState();
      if (st.ok) {
        this._applyState(st.data || {});
      } else {
        this._applyState(r.data || {});
      }
      if (S.track.loaded) {
        this.seek(0, true);
        const diag = S.track.diagnose || {};
        toast('ok', '日志已导入', (diag.count || 0) + ' 个定位点 · ' + (diag.stamp_kind_zh || ''));
      } else {
        toast('warn', '没有解析出定位点', '看回执里「认不出的行」对照格式');
      }
      this.render();
    },

    /* ------------------------------------------------------ 取帧（纯本地） */
    /**
     * 视频时刻 → 定位帧。**纯计算，不发任何请求。**
     *
     * 整条序列在导入时一次性下发（``S.track.scrub``），这里只做二分 + 线性插值。
     * 逐帧问后端是行不通的：拖动时间轴一秒能触发 60 次请求，实测会把浏览器的
     * 连接池拖死 —— 请求排队、全部等到 120 s 客户端超时，页面看起来就是"卡住"。
     *
     * 对齐锚点与后端 ``at_video_time`` 同一套规则：未对齐时退化为
     * 「视频 0 秒 = 日志起点」，并把 ``aligned=false`` 带出去供界面提示。
     */
    frameAt(tVideo) {
      const s = S.track.summary;
      const ser = S.track.scrub;
      if (!s || !ser || !ser.length) return null;
      const origin = (s.origin_s != null) ? +s.origin_s : (s.t0 != null ? +s.t0 : null);
      if (origin == null) return null;
      return this._sample(ser, origin + (+tVideo || 0), !!s.aligned);
    },

    /** 在序列里取 ``tLog`` 时刻的一帧。连续量插值，离散量取更近的一端。 */
    _sample(ser, tLog, aligned) {
      const n = ser.length;
      let i;
      if (tLog <= ser[0].t) i = 0;
      else if (tLog >= ser[n - 1].t) i = n - 1;
      else {
        // 二分而不是线性扫描：序列最多 1.2 万点，回放时每秒要调用 60 次，
        // 线性扫描在这个量级已经能感觉到掉帧。
        let lo = 0, hi = n - 1;
        while (lo + 1 < hi) {
          const mid = (lo + hi) >> 1;
          if (ser[mid].t <= tLog) lo = mid; else hi = mid;
        }
        i = lo;
      }
      const a = ser[i];
      const b = (i + 1 < n) ? ser[i + 1] : null;
      const out = Object.assign({}, a);
      out.t = tLog;
      out.aligned = aligned;
      delete out.sat;
      out.satellites = (a.sat == null) ? [] : (S.track.scrubSats[a.sat] || []);

      if (!b || b.t <= a.t || tLog <= a.t) { out.interpolated = false; return out; }

      const k = (tLog - a.t) / (b.t - a.t);
      // 只插连续量。定位质量、卫星数是离散的，插值出来是**假数据**
      // （"HDOP 1.4 和 1.8 之间"没有意义，"定位质量 1.5" 更是无中生有）。
      ['lat', 'lon', 'alt', 'speed_kmh', 'course', 'hdop', 'vdop', 'pdop'].forEach(key => {
        const va = a[key], vb = b[key];
        if (typeof va === 'number' && typeof vb === 'number') {
          out[key] = +(va + (vb - va) * k).toFixed(7);
        }
      });
      const near = (k < 0.5) ? a : b;
      ['fix_quality', 'fix_quality_zh', 'satellites_used', 'satellites_visible',
        'utc', 'usable', 'mode_zh'].forEach(key => { out[key] = near[key]; });
      if (near === b) {
        out.satellites = (b.sat == null) ? [] : (S.track.scrubSats[b.sat] || []);
      }
      out.interpolated = true;
      return out;
    },

    /** 拖到某一时刻：取帧 + 刷新所有跟着时间走的视图。**同步**，不再有 await。 */
    seek(t, silent) {
      const s = S.track.summary;
      if (!S.track.loaded || !s) return;
      const dur = +s.duration_s || 0;
      t = Math.max(0, Math.min(dur, +t || 0));
      S.track.t = t;
      S.track.fix = this.frameAt(t);
      const sl = $('#tl-slider');
      if (sl && +sl.value !== t) sl.value = String(t);
      const now = $('#tl-now');
      if (now) now.textContent = t.toFixed(1) + ' s';
      if (!silent) this.renderFix();
      if (S.view === 'geo') { Geo.renderSky(); Geo.renderScatter(); }
    },

    play() {
      if (this.timer) { this.stop(); return; }
      const s = S.track.summary;
      if (!S.track.loaded || !s) return;
      const dur = +s.duration_s || 0;
      if (dur <= 0) return;
      if (S.track.t >= dur - 1e-6) S.track.t = 0;
      let last = performance.now();
      const btn = $('#btn-tl-play');
      if (btn) btn.innerHTML = '<svg><use href="#i-stop"/></svg>暂停';

      // 序列在本地之后，一帧就是一次纯计算，中间没有 await —— 也就不存在
      // "暂停恰好落在一帧的 await 期间、那一帧醒来又把循环续上"的窗口了。
      // 代际号保留：它是 stop() 与 rAF 之间唯一的同步点，成本为零。
      const gen = ++this._gen;
      const tick = () => {
        if (gen !== this._gen) return;
        const now = performance.now();
        const dt = (now - last) / 1000;
        last = now;
        S.track.t += dt;                    // 1× 实时速度
        if (S.track.t >= dur) { this.seek(dur); this.stop(); return; }
        this.seek(S.track.t, true);
        this.timer = requestAnimationFrame(tick);
      };
      this.timer = requestAnimationFrame(tick);
    },

    stop() {
      this._gen = (this._gen || 0) + 1;
      if (this.timer) { cancelAnimationFrame(this.timer); this.timer = null; }
      const btn = $('#btn-tl-play');
      if (btn) btn.innerHTML = '<svg><use href="#i-play"/></svg>播放';
    },

    async applyAlign(payload) {
      const r = await Api.trackAlign(payload);
      const d = r.data || {};
      if (!d.ok) { toast('bad', '对齐失败', d.error || r.error || '未知错误'); return; }
      S.track.summary = d.summary || S.track.summary;
      const used = { utc: '视频起始时刻', origin_s: '日志时刻', delta_s: '整体平移', reset: '已取消对齐' }[d.used] || d.used;
      toast('ok', '对齐已更新', '依据：' + used);
      // 序列本身没变，变的只是锚点 —— 本地重算即可，不必再拉一次数据
      this.seek(S.track.t, true);
      this.render();
      if (S.view === 'geo') { Geo.renderSky(); Geo.renderScatter(); }
    },

    async clear() {
      this.stop();
      await Api.trackClear();
      S.track = Object.assign(JSON.parse(JSON.stringify(BDS.DEFAULTS.track)));
      this.render();
      if (S.view === 'geo') { Geo.renderSky(); Geo.renderScatter(); }
    },

    /** 紧凑的单行定位读数（视频进度条下方那条） */
    fixLine(f, t) {
      if (!f) {
        return '<p class="fld-hint" style="padding:8px 0 0">视频 ' + (+t || 0).toFixed(1) +
          ' s 处没有对应的定位点（日志覆盖不到这一段）。</p>';
      }
      const ok = !!f.usable && f.lat != null && f.lon != null;
      const chipCls = ok ? 'ok' : 'bad';
      const chipTxt = ok ? (f.fix_quality_zh || '已定位') : (f.fix_quality_zh || '无定位');
      const kv = (k, v) => '<span class="k">' + k + '</span><b>' + v + '</b>';
      return '<div class="fixline">' +
        '<span class="chip ' + chipCls + '"><span class="dot"></span>' + U.esc(chipTxt) + '</span>' +
        (f.aligned === false ? '<span class="chip warn"><span class="dot"></span>未对齐</span>' : '') +
        kv('视频时刻', nz(t, 1, ' s')) +
        kv('UTC', U.esc(f.utc || '—')) +
        kv('LAT', ok ? Number(f.lat).toFixed(6) : '—') +
        kv('LON', ok ? Number(f.lon).toFixed(6) : '—') +
        kv('卫星', (f.satellites_used == null ? '—' : f.satellites_used) + '/' +
          (f.satellites_visible == null ? '—' : f.satellites_visible)) +
        kv('HDOP', nz(f.hdop, 2)) +
        '</div>';
    },

    /* ------------------------------------------------------ 渲染 */
    render() {
      const t = S.track;
      const chip = $('#track-chip');
      if (chip) {
        const diag = t.diagnose || {};
        if (t.loading) { chip.className = 'chip'; chip.textContent = '解析中…'; }
        else if (!t.loaded) {
          chip.className = 'chip ' + (diag.count === 0 && diag.hints ? 'bad' : '');
          chip.textContent = diag.hints ? '解析失败' : '未导入';
        } else {
          // 相对时间不能显示成"已对齐"，否则用户以为时间轴是准的
          const abs = t.summary && t.summary.absolute;
          chip.className = 'chip ' + (abs ? 'ok' : 'warn');
          chip.textContent = abs ? '已导入 · 绝对时间' : '已导入 · 相对时间';
        }
      }
      const clr = $('#btn-track-clear');
      if (clr) clr.hidden = !t.loaded && !t.diagnose;

      this.renderReceipt();
      const tl = $('#track-tl');
      const s = t.summary;
      if (tl) tl.hidden = !(t.loaded && s);
      if (t.loaded && s) {
        const dur = +s.duration_s || 0;
        const sl = $('#tl-slider');
        if (sl) { sl.max = String(dur); sl.value = String(Math.min(t.t, dur)); }
        const range = $('#tl-range');
        if (range) range.textContent = '0 ~ ' + dur.toFixed(1) + ' s · ' + (s.count || 0) + ' 个定位点';
        const now = $('#tl-now');
        if (now) now.textContent = (+t.t || 0).toFixed(1) + ' s';
        const al = $('#tl-aligned');
        if (al) {
          al.textContent = s.aligned ? ('已对齐 · 平移 ' + (+s.shift_s || 0).toFixed(1) + ' s') : '未对齐（按起点回放）';
        }
      }
      const alBox = $('#track-align');
      if (alBox) alBox.hidden = !(t.loaded && s);
      this.renderFix();
    },

    /** 导入回执：识别到了什么、哪些没认出来、下一步该干什么 */
    renderReceipt() {
      const host = $('#track-receipt');
      if (!host) return;
      const t = S.track;
      const d = t.diagnose;
      if (!d) {
        host.innerHTML = t.loaded ? '' :
          '<p class="fld-hint" style="padding:0">还没有导入日志。导入后这里会显示识别结果与格式诊断。</p>';
        return;
      }
      const ok = !!d.ok;
      const rows = [
        ['文件名', t.name || '—'],
        ['识别到的格式', d.stamp_kind_zh || '—'],
        ['定位点', d.count != null ? String(d.count) : '—'],
        ['时长', d.duration_s != null ? (+d.duration_s).toFixed(1) + ' s' : '—'],
        ['历元间隔', d.epoch_interval_s != null ? (+d.epoch_interval_s).toFixed(2) + ' s' : '—'],
        // 时间范围是长值，独占一行：塞进 140px 的格子里会被折成四行还从中间断词
        ['时间范围', Track.timeRange(d.first_utc, d.last_utc), true],
        ['总行数', d.total_lines != null ? String(d.total_lines) : '—'],
        ['注释 / 空行', d.comment_lines != null ? String(d.comment_lines) : '—'],
        ['未识别', d.skipped_lines != null ? String(d.skipped_lines) : '—'],
      ];
      if (d.clock_offset_s != null && Math.abs(d.clock_offset_s) >= 1) {
        rows.push(['MCU 时钟偏移', (+d.clock_offset_s).toFixed(1) + ' s']);
      }
      let html = '<div class="rcpt ' + (ok ? 'hd-ok' : 'hd-bad') + '">' +
        '<div class="rcpt-hd"><svg><use href="' + (ok ? '#i-check' : '#i-alert') + '"/></svg>' +
        (ok ? '已解析出 ' + d.count + ' 个定位点' : '没有解析出定位点') + '</div>' +
        '<div class="rcpt-grid">' + rows.map(r =>
          '<div class="meta-i' + (r[2] ? ' wide' : '') + '"><span class="k">' + U.esc(r[0]) + '</span>' +
          '<span class="v">' + U.esc(String(r[1])) + '</span></div>').join('') + '</div>';

      if (d.hints && d.hints.length) {
        html += '<div class="rcpt-notes"><ul>' +
          d.hints.map(h => '<li>' + U.esc(h) + '</li>').join('') + '</ul></div>';
      }
      if (d.skipped_samples && d.skipped_samples.length) {
        html += '<div class="rcpt-code">认不出的行：\n' +
          d.skipped_samples.map(s => U.esc(s)).join('\n') + '</div>';
      } else if (d.sample_lines && d.sample_lines.length && !ok) {
        html += '<div class="rcpt-code">文件头几行：\n' +
          d.sample_lines.map(s => U.esc(s)).join('\n') + '</div>';
      }
      host.innerHTML = html + '</div>';
    },

    /**
     * 时间范围的紧凑写法。同一天就不重复日期 ——
     * ``2026-09-17T12:00:00Z → 2026-09-17T12:00:19Z`` 里那 11 个字符的日期
     * 出现了两遍，读的人还要自己去比对是不是同一天。
     */
    timeRange(a, b) {
      if (!a || !b) return '（相对时间，无绝对范围）';
      const short = s => String(s).replace(/^(\d{4})-(\d{2})-(\d{2})T/, '$1-$2-$3 ');
      if (a.slice(0, 10) === b.slice(0, 10)) {
        return short(a) + ' → ' + String(b).slice(11);
      }
      return short(a) + ' → ' + short(b);
    },

    /** 当前时刻的定位帧 */
    renderFix() {
      const host = $('#track-fix');
      if (!host) return;
      if (!S.track.loaded) { host.innerHTML = ''; return; }
      const f = S.track.fix;
      if (!f) {
        host.innerHTML = '<p class="fld-hint" style="padding:0">该时刻没有定位点。</p>';
        return;
      }
      const ok = !!f.usable && f.lat != null && f.lon != null;
      const tile = (k, v, d, cls) =>
        '<div class="tile"><span class="k">' + k + '</span>' +
        '<span class="v ' + (cls || '') + '">' + v + '</span>' +
        '<span class="d">' + d + '</span></div>';
      // UTC 是**语句自带的**（历元粒度，1 s 或 0.1 s），不能拿插值出来的
      // 日志时刻去重算 —— 两者在"MCU 按本地时区写时间戳"时会差好几个小时，
      // 用 f.t 重算等于把本地时间标成 UTC。所以这里只把粒度讲清楚：
      // 经纬度是插值的，UTC 是最近历元的。不说明的话，用户看到
      // 「视频时刻 12.4 s / UTC 12:00:12Z」会以为是程序算错了。
      const utcDesc = (f.interpolated ? '最近历元 · ' : '') +
        (ok ? (f.fix_quality_zh || '') : '无定位');
      host.innerHTML =
        '<div class="tiles c4">' +
        // 显示**视频时刻**而不是 f.t：f.t 是日志时间轴上的值，绝对时间日志下
        // 它就是个 Unix 秒（1789646415.4），摆在界面上没人看得懂。
        tile('视频时刻', nz(S.track.t, 1, ' s'), f.interpolated ? '插值得到' : '历元原始值') +
        tile('UTC', f.utc || '—', utcDesc) +
        tile('纬度 LAT', ok ? Number(f.lat).toFixed(6) : '—', 'WGS-84', 'geo') +
        tile('经度 LON', ok ? Number(f.lon).toFixed(6) : '—', 'WGS-84', 'geo') +
        tile('海拔', ok && f.alt != null ? Number(f.alt).toFixed(1) + '<small>m</small>' : '—', '椭球高') +
        tile('卫星 已用/可见',
          (f.satellites_used == null ? '—' : f.satellites_used) + ' / ' +
          (f.satellites_visible == null ? '—' : f.satellites_visible), '颗') +
        tile('HDOP', nz(f.hdop, 2), '越小越好') +
        tile('速度 / 航向', nz(f.speed_kmh, 1, ' km/h') + ' · ' + nz(f.course, 0, '°'), 'RMC') +
        '</div>' +
        (ok ? '' : '<p class="fld-hint" style="padding:6px 0 0">该时刻没有有效定位解（' +
          U.esc(f.fix_quality_zh || '未知') + '）。</p>');
    },

    /* ------------------------------------------------------ 事件 */
    bind() {
      const dz = $('#track-drop');
      const fi = $('#track-file');
      if (!dz || !fi) return;
      dz.onclick = () => fi.click();
      dz.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fi.click(); } };
      fi.onchange = () => { if (fi.files && fi.files[0]) this.importFile(fi.files[0]); fi.value = ''; };

      ['dragenter', 'dragover'].forEach(ev => {
        dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('over'); });
      });
      ['dragleave', 'drop'].forEach(ev => {
        dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('over'); });
      });
      dz.addEventListener('drop', e => {
        const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) this.importFile(f);
      });

      const sl = $('#tl-slider');
      if (sl) {
        // input 事件在拖动中连续触发 → 取帧要不插值地快，用静默模式避免整块重绘
        sl.oninput = e => { this.stop(); this.seek(+e.target.value, false); };
      }
      const play = $('#btn-tl-play');
      if (play) play.onclick = () => this.play();
      const st = $('#btn-tl-start');
      if (st) st.onclick = () => { this.stop(); this.seek(0); };
      const en = $('#btn-tl-end');
      if (en) en.onclick = () => {
        this.stop();
        this.seek(S.track.summary ? +S.track.summary.duration_s || 0 : 0);
      };
      const bu = $('#btn-al-utc');
      if (bu) bu.onclick = () => {
        const v = ($('#al-utc').value || '').trim();
        if (!v) { toast('warn', '请填写 UTC 时刻', '形如 2026-09-17T12:00:00Z'); return; }
        this.applyAlign({ utc: v });
      };
      const bd = $('#btn-al-delta');
      if (bd) bd.onclick = () => this.applyAlign({ delta_s: +($('#al-delta').value || 0) });
      const clr = $('#btn-track-clear');
      if (clr) clr.onclick = () => this.clear();
    },
  };

  BDS.Track = Track;             // 暴露给 dev 探针与调试用

  /* ======================================================== 设置视图 */
  const Settings = {
    render() {
      $$('[data-theme-opt]').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.themeOpt === S.theme)));
      $$('[data-pal]').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.pal === S.palette)));
      $$('#set-density .seg-btn').forEach(b => b.setAttribute('aria-selected', String(b.dataset.v === S.density)));
      setSwitch($('#set-motion'), S.motion);
      setSwitch($('#set-autoprobe'), S.api.autoprobe);
      setSwitch($('#set-persist'), S.persist);
      setSwitch($('#set-slice'), S.params.slice);
      $('#set-api').value = S.api.base || '';
      $('#set-timeout').value = S.api.timeout;
      $('#set-conf').value = S.params.conf;
      $('#set-conf-v').textContent = (+S.params.conf).toFixed(2);
      $('#set-iou').value = S.params.iou;
      $('#set-iou-v').textContent = (+S.params.iou).toFixed(2);
      $('#set-imgsz').value = String(S.params.imgsz);
      const rng = $('#set-conf'); rng.style.setProperty('--p', (rng.value * 100) + '%');
      const rng2 = $('#set-iou'); rng2.style.setProperty('--p', (rng2.value * 100) + '%');
      // 只刷会话统计那一块（`#data-stat` 在设置页里）。训练指标已经搬到分析视图，
      // 由 Analytics.render() 自己负责加载 —— 这里再调一次会白跑一趟请求。
      Analytics.renderSession();
    },
  };

  function applyTheme(theme) {
    S.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    $('#btn-theme').innerHTML = '<svg><use href="#' + (theme === 'dark' ? 'i-moon' : 'i-sun') + '"/></svg>';
    BDS.prefs.save();
    renderStage();
    renderRight();
    if (S.view === 'analytics') Analytics.render();
    if (S.view === 'geo') Geo.render();
  }

  function applyDensity(d) {
    S.density = d;
    document.documentElement.setAttribute('data-density', d);
    BDS.prefs.save();
  }

  function applyMotion(on) {
    S.motion = on;
    document.documentElement.setAttribute('data-motion', on ? 'on' : 'off');
    BDS.prefs.save();
  }

  function applyPalette(p) {
    S.palette = p;
    document.documentElement.setAttribute('data-pal', p);
    BDS.prefs.save();
    renderRight(); renderStage();
    if (S.view === 'analytics') Analytics.render();
    if (S.view === 'geo') Geo.render();
  }

  /* ======================================================== 命令面板 */
  const Commands = [
    { g: '视图', id: 'v-detect', t: '检测台', d: '上传与推理工作台', i: 'i-scan', run: () => go('detect') },
    { g: '视图', id: 'v-jobs', t: '任务', d: '视频推理任务列表', i: 'i-list', run: () => go('jobs') },
    { g: '视图', id: 'v-analytics', t: '分析', d: '会话统计与图表', i: 'i-chart', run: () => go('analytics') },
    { g: '视图', id: 'v-geo', t: '北斗定位', d: '星空图与目标地理散布', i: 'i-globe', run: () => go('geo') },
    { g: '视图', id: 'v-settings', t: '设置', d: '外观 / 连接 / 默认参数', i: 'i-settings', run: () => go('settings') },
    { g: '视图', id: 'v-train', t: '训练成果', d: '真实训练指标、收敛曲线与消融对比', i: 'i-activity', run: () => { go('settings'); setTimeout(() => { const el = $('#train-kpis'); if (el) el.scrollIntoView({ block: 'center', behavior: 'smooth' }); }, 140); } },

    { g: '操作', id: 'a-demo', t: '载入演示场景', d: '生成一幅航拍场景并附带检测结果', i: 'i-sparkles', run: () => { go('detect'); loadDemo(true); } },
    { g: '操作', id: 'a-run', t: '开始推理', d: '对当前输入执行检测', i: 'i-play', run: () => { go('detect'); runInference(); } },
    { g: '操作', id: 'a-upload', t: '上传图片', d: '打开文件选择器', i: 'i-upload', run: () => $('#file-input').click() },
    { g: '操作', id: 'a-video', t: '上传视频', d: '提交视频推理任务', i: 'i-film', run: () => { S.source = 'video'; syncSource('video'); go('detect'); $('#file-video').click(); } },
    { g: '操作', id: 'a-fit', t: '适应窗口', d: '重置缩放与平移', i: 'i-fit', run: () => { Overlay.fit(); renderStage(); } },
    { g: '操作', id: 'a-export-json', t: '导出 JSON', d: '当前帧检测结果', i: 'i-download', run: () => exportResults('json') },
    { g: '操作', id: 'a-export-csv', t: '导出 CSV', d: '当前帧检测结果表格', i: 'i-table', run: () => exportResults('csv') },
    { g: '操作', id: 'a-export-png', t: '导出标注图', d: '合成带框 PNG', i: 'i-image', run: () => exportResults('png') },
    { g: '操作', id: 'a-clear-queue', t: '清空队列', d: '移除全部已载入文件', i: 'i-trash', run: () => clearQueue() },

    { g: '外观', id: 'p-dark', t: '切换深色主题', d: '暗光值守', i: 'i-moon', run: () => applyTheme('dark') },
    { g: '外观', id: 'p-light', t: '切换浅色主题', d: '投影与打印', i: 'i-sun', run: () => applyTheme('light') },
    { g: '外观', id: 'p-soft', t: '配色：柔和', d: '默认类别配色', i: 'i-sparkles', run: () => applyPalette('soft') },
    { g: '外观', id: 'p-vivid', t: '配色：鲜明', d: '高饱和类别配色', i: 'i-sparkles', run: () => applyPalette('vivid') },
    { g: '外观', id: 'p-mono', t: '配色：单色', d: '单色阶类别配色', i: 'i-sparkles', run: () => applyPalette('mono') },

    { g: '连接', id: 'c-test', t: '测试后端连接', d: '探测 /api/v1 与 /api', i: 'i-plug', run: () => testConnection(true) },
    { g: '连接', id: 'c-doc', t: '打开接口文档', d: '后端 OpenAPI 页面', i: 'i-terminal', run: () => window.open(Api.url('/docs'), '_blank') },

    { g: '帮助', id: 'h-keys', t: '快捷键', d: '查看全部键盘操作', i: 'i-keyboard', run: () => openModal('modal-keys') },
    { g: '帮助', id: 'h-about', t: '关于本系统', d: '模型改进与设计参考', i: 'i-info', run: () => openModal('modal-about') },
    { g: '帮助', id: 'h-clear', t: '清空会话统计', d: '重置分析视图数据', i: 'i-trash', run: () => { clearSession(); Analytics.render(); toast('info', '会话统计已清空'); } },
  ];

  let palItems = [], palSel = 0;
  function openPalette() {
    $('#palette').classList.add('is-open');
    $('#pal-input').value = '';
    filterPalette('');
    setTimeout(() => $('#pal-input').focus(), 20);
  }
  function closePalette() { $('#palette').classList.remove('is-open'); }

  function filterPalette(q) {
    palItems = Commands.map(c => ({ c: c, s: Math.max(U.fuzzy(q, c.t) * 1.4, U.fuzzy(q, c.d), U.fuzzy(q, c.g) * 0.6) }))
      .filter(x => x.s > 0)
      .sort((a, b) => b.s - a.s)
      .map(x => x.c);
    palSel = 0;
    renderPalette();
  }

  function renderPalette() {
    const host = $('#pal-list');
    if (!palItems.length) {
      host.innerHTML = '<div class="pal-empty">没有匹配的命令</div>';
      return;
    }
    let html = '', lastG = null;
    palItems.forEach((c, i) => {
      if (c.g !== lastG) { html += '<div class="pal-group">' + U.esc(c.g) + '</div>'; lastG = c.g; }
      html += '<div class="pal-item" data-i="' + i + '" aria-selected="' + (i === palSel) + '">' +
        '<svg><use href="#' + c.i + '"/></svg><span>' + U.esc(c.t) + '</span>' +
        '<div class="sp"></div><span class="d">' + U.esc(c.d) + '</span></div>';
    });
    host.innerHTML = html;
    U.$$('.pal-item', host).forEach(el => {
      el.onclick = () => runCommand(+el.dataset.i);
      el.onmouseenter = () => { palSel = +el.dataset.i; renderPalette(); };
    });
    const cur = host.querySelector('[aria-selected="true"]');
    if (cur) cur.scrollIntoView({ block: 'nearest' });
  }

  function runCommand(i) {
    const c = palItems[i];
    if (!c) return;
    closePalette();
    setTimeout(() => c.run(), 30);
  }

  /* ======================================================== 弹窗 */
  function openModal(id) { $('#' + id).classList.add('is-open'); }
  function closeModal(id) { $('#' + id).classList.remove('is-open'); }
  function closeAllModals() { $$('.modal').forEach(m => m.classList.remove('is-open')); }

  /* ======================================================== 连接测试 */
  async function testConnection(notify) {
    const el = $('#conn-result');
    if (el) el.textContent = '正在探测…';
    const r = await Api.probe();
    if (r.online) {
      const s = await Api.status();
      if (s.ok) applyStatus(s);
      Train.load(true);   // 连上后端就重新拉一次训练指标
      ModelSelector.refresh();   // 同步权重候选与当前选中项
      if (el) el.textContent = '已连接 · ' + (r.shape === 'fastapi' ? 'FastAPI' : 'Flask') + ' ' + r.prefix + ' · ' + S.model.mode;
      if (notify) toast('ok', '连接成功', '后端 ' + (r.shape === 'fastapi' ? 'FastAPI' : 'Flask') + ' · 模式 ' + S.model.mode);
    } else {
      if (el) el.textContent = '未检测到后端服务（已尝试 /api/v1 与 /api）';
      if (notify) toast('warn', '未检测到后端', '页面仍可完整演示，启动后端后点此重试');
    }
    renderStatus();
    if (S.view === 'settings') Settings.render();
    return r.online;
  }

  function applyStatus(s) {
    const d = s.data;
    S.model.mode = d.mode;
    S.model.device = d.device;
    S.model.cudaName = d.cudaName;
    S.model.weights = d.weights;
    S.model.weightsLabel = d.weights ? ModelSelector.label(d.weights) : null;
    S.model.loaded = d.loaded;
    S.model.engineAvailable = d.engineAvailable;
    S.model.slicingAvailable = d.slicingAvailable;
    S.model.message = d.message;
    if (d.beidou) applyBeidou(d.beidou);
    if (d.message && d.mode !== 'real') showBanner('warn', d.message);
    renderStatus();
  }

  /* ======================================================== 模型选择器 */
  /**
   * 顶栏权重下拉：候选来自 `GET /models`，切换走 `POST /models/switch`
   * （进程内原地替换，不重启后端）。只有真实存在的权重可选，缺失的列出但禁用，
   * 便于直观看出探测路径。
   */
  const ModelSelector = {
    busy: false,
    loaded: null,
    exist: [],

    el() { return $('#model-select'); },

    /** 绝对路径 → 「训练产物名 / 文件名」短标签。
     *  训练产物是 <run>/weights/best.pt，直接用父目录会得到 "weights / best.pt"，
     *  导致 v1/v4 两个选项显示成同一个名字 —— 所以要再往上取一层。 */
    label(p) {
      if (!p) return '未加载';
      const parts = String(p).split(/[\\/]/);
      const file = parts.pop() || String(p);
      let run = parts.pop() || '';
      if (run.toLowerCase() === 'weights') run = parts.pop() || run;
      return run ? run + ' / ' + file : file;
    },

    async refresh() {
      const el = this.el();
      if (!el) return;
      if (!S.api.online) {
        el.disabled = true;
        el.innerHTML = '<option>未连接后端</option>';
        return;
      }
      const r = await Api.models();
      if (!r.ok) {
        el.disabled = true;
        el.innerHTML = '<option>不可用</option>';
        return;
      }
      const d = r.data;
      this.exist = d.candidates_exist || [];
      this.loaded = d.loaded_variant || null;

      // 按「标签」去重：例如 C:/yolo_runs/train/visdrone_v4/weights/best.pt 与
      // 项目内 runs/train/visdrone_v4/weights/best.pt 标签相同，前者存在后者缺失，
      // 两个同名条目同时出现在下拉框里很迷惑。只保留第一个（存在的优先）。
      const seen = new Set();
      const ordered = [];
      const pushUnique = (p) => {
        const lab = this.label(p);
        if (seen.has(lab)) return;
        seen.add(lab);
        ordered.push(p);
      };
      this.exist.forEach(pushUnique);
      (d.candidates || []).forEach((p) => {
        if (this.exist.indexOf(p) < 0) pushUnique(p);
      });

      el.innerHTML = ordered.map(p => {
        const ok = this.exist.indexOf(p) >= 0;
        return '<option value="' + U.esc(p) + '"' + (ok ? '' : ' disabled') + '>' +
          U.esc(this.label(p)) + (ok ? '' : '（缺失）') + '</option>';
      }).join('');
      el.value = this.loaded || (this.exist[0] || '');
      el.disabled = this.busy || !this.exist.length;
      el.title = this.loaded ? ('当前加载：' + this.loaded) : '选择要加载的模型权重';
      // 暴露当前权重标签给顶栏芯片渲染（避免 v4/v1 都显示成 "best.pt"）
      this.loadedLabel = this.loaded ? this.label(this.loaded) : null;
    },

    async switchTo(weight) {
      const el = this.el();
      if (!el || this.busy || !weight) return;
      if (weight === this.loaded) return;
      if (!window.confirm(
        '切换模型会在后端重新加载权重（约十几秒，期间该请求会阻塞）。\n\n' +
        '目标：' + this.label(weight) + '\n当前：' + this.label(this.loaded)
      )) {
        el.value = this.loaded || '';   // 取消 → 回显当前
        return;
      }

      this.busy = true;
      el.disabled = true;
      el.value = weight;
      toast('ok', '正在切换模型…', '加载权重中，请稍候');

      const r = await Api.switchModel(weight);
      this.busy = false;

      if (!r.ok) {
        toast('bad', '模型切换失败', r.error || '未知错误', 6000);
        await this.refresh();
        return;
      }

      this.loaded = weight;
      const s = await Api.status();
      if (s.ok) applyStatus(s); else renderStatus();
      await this.refresh();
      toast('ok', '模型已切换', '当前：' + this.label(weight));
    },
  };

  /* ======================================================== 摄像头 */
  let camStream = null, camTimer = null, camBusy = false;
  async function startCam() {
    try {
      camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 } } });
    } catch (e) {
      toast('bad', '无法访问摄像头', String(e && e.message ? e.message : e));
      return;
    }
    const id = 'cam';
    let f = S.frames[id];
    if (!f) {
      const v = document.createElement('video');
      v.autoplay = true; v.muted = true; v.playsInline = true;
      f = S.frames[id] = {
        id: id, name: '摄像头 · LIVE', size: 0, type: 'cam', url: null, thumb: null,
        img: v, width: 0, height: 0, dets: [], done: false, demo: false,
      };
      S.queue.push({ id: id, name: '摄像头 · LIVE', size: 0, type: 'cam', url: null, thumb: null });
    }
    f.img.srcObject = camStream;
    await f.img.play();
    await new Promise(r => setTimeout(r, 300));
    f.width = f.img.videoWidth || 1280;
    f.height = f.img.videoHeight || 720;
    S.camOn = true;
    setActive('cam');
    showStage(f);
    setSwitch($('#p-slice'), S.params.slice);
    $('#btn-cam-start').disabled = true; $('#btn-cam-stop').disabled = false;
    renderQueue();
    toast('ok', '摄像头已启动', S.api.online ? '正在逐帧推理' : '未连接后端，仅预览画面');
    camTimer = setInterval(camTick, 900);
    camTick();
  }

  async function camTick() {
    if (camBusy || !S.camOn) return;
    const f = S.frames['cam'];
    if (!f || !f.img.videoWidth) return;
    camBusy = true;
    f.width = f.img.videoWidth; f.height = f.img.videoHeight;
    if (S.api.online) {
      try {
        const cv = document.createElement('canvas');
        const k = Math.min(1, 960 / f.width);
        cv.width = Math.round(f.width * k); cv.height = Math.round(f.height * k);
        cv.getContext('2d').drawImage(f.img, 0, 0, cv.width, cv.height);
        const data = cv.toDataURL('image/jpeg', 0.72);
        const r = await Api.detect(data, S.params);
        if (r.ok) {
          f.dets = r.data.detections.map((d, i) => { d._i = i; return d; });
          f.ms = r.data.elapsedMs != null ? r.data.elapsedMs : r.ms;
          f.device = r.data.device; f.engine = r.data.engine || 'live';
          f.done = true;
          setReqId(r.requestId);
          renderStage(); renderRight(); updateHUD(f); setLatency(f.ms);
        }
      } catch (e) { /* 丢帧 */ }
    } else {
      f.dets = []; f.ms = null; f.done = true;
      renderStage();
    }
    camBusy = false;
  }

  function stopCam() {
    if (camTimer) { clearInterval(camTimer); camTimer = null; }
    if (camStream) { camStream.getTracks().forEach(t => t.stop()); camStream = null; }
    S.camOn = false;
    const f = S.frames['cam'];
    if (f) f.img.srcObject = null;
    $('#btn-cam-start').disabled = false; $('#btn-cam-stop').disabled = true;
    toast('info', '摄像头已停止');
  }

  /* ======================================================== 源切换 */
  function syncSource(src) {
    S.source = src;
    $$('#seg-source .seg-btn').forEach(b => b.setAttribute('aria-selected', String(b.dataset.src === src)));
    $$('[data-src-panel]').forEach(p => { p.hidden = p.dataset.srcPanel !== src; });
    if (src !== 'cam' && S.camOn) stopCam();
  }

  /* ======================================================== 绑定 */
  function bindStage() {
    const stage = $('#stage');
    let dragging = false, moved = false, lx = 0, ly = 0;

    stage.addEventListener('wheel', e => {
      if (!Overlay.img) return;
      e.preventDefault();
      const rect = stage.getBoundingClientRect();
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      Overlay.zoomAt(e.clientX - rect.left, e.clientY - rect.top, factor);
      S.autoFit = false;
      $('#zoom-val').textContent = Math.round(Overlay.view.scale * 100) + '%';
    }, { passive: false });

    stage.addEventListener('pointerdown', e => {
      if (!Overlay.img) return;
      if (e.button !== 0) return;
      dragging = true; moved = false;
      lx = e.clientX; ly = e.clientY;
      stage.classList.add('is-panning');
      stage.setPointerCapture(e.pointerId);
    });

    stage.addEventListener('pointermove', e => {
      const rect = stage.getBoundingClientRect();
      if (dragging) {
        const dx = e.clientX - lx, dy = e.clientY - ly;
        if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
        if (moved) {
          Overlay.panBy(dx, dy);
          S.autoFit = false;
          lx = e.clientX; ly = e.clientY;
        }
      }
      if (Overlay.img) {
        updateHUDCoords(e.clientX - rect.left, e.clientY - rect.top);
        const hit = Overlay.hit(e.clientX - rect.left, e.clientY - rect.top);
        if (hit !== S.hover) {
          S.hover = hit;
          stage.style.cursor = hit >= 0 ? 'pointer' : '';
          renderStage();
        }
      }
    });

    stage.addEventListener('pointerup', e => {
      stage.classList.remove('is-panning');
      if (!dragging) return;
      dragging = false;
      if (!moved && Overlay.img) {
        const rect = stage.getBoundingClientRect();
        const hit = Overlay.hit(e.clientX - rect.left, e.clientY - rect.top);
        S.selected = hit;
        renderRight(); renderStage();
      }
    });

    stage.addEventListener('pointerleave', () => {
      if (S.hover >= 0) { S.hover = -1; renderStage(); }
    });

    stage.addEventListener('dblclick', e => {
      if (!Overlay.img) return;
      const rect = stage.getBoundingClientRect();
      const hit = Overlay.hit(e.clientX - rect.left, e.clientY - rect.top);
      if (hit >= 0) { S.selected = hit; Overlay.focusOn(hit); renderRight(); renderStage(); }
    });
  }

  function bindUI() {
    // 图标栏
    $$('.rail-btn[data-view]').forEach(b => { b.onclick = () => go(b.dataset.view); });
    $('#rail-logo').onclick = () => go('detect');
    $('#rail-keys').onclick = () => openModal('modal-keys');
    $('#rail-about').onclick = () => openModal('modal-about');

    // 顶栏
    $('#cmd-open').onclick = openPalette;
    $('#btn-theme').onclick = () => applyTheme(S.theme === 'dark' ? 'light' : 'dark');

    // 命令面板
    $('#palette').addEventListener('mousedown', e => { if (e.target.id === 'palette') closePalette(); });
    $('#pal-input').addEventListener('input', e => filterPalette(e.target.value));
    $('#pal-input').addEventListener('keydown', e => {
      if (e.key === 'ArrowDown') { e.preventDefault(); palSel = Math.min(palItems.length - 1, palSel + 1); renderPalette(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); palSel = Math.max(0, palSel - 1); renderPalette(); }
      else if (e.key === 'Enter') { e.preventDefault(); runCommand(palSel); }
      else if (e.key === 'Escape') { closePalette(); }
    });

    // 弹窗
    $$('[data-close]').forEach(b => { b.onclick = () => closeModal(b.dataset.close); });
    $$('.modal').forEach(m => m.addEventListener('mousedown', e => { if (e.target === m) closeModal(m.id); }));
    $('#btn-keys2').onclick = () => openModal('modal-keys');
    $('#btn-keys3').onclick = () => openModal('modal-keys');
    $('#btn-api-doc').onclick = () => window.open(Api.url('/docs'), '_blank');

    // 横幅
    $('#banner-x').onclick = hideBanner;

    // 输入源
    bindSeg($('#seg-source'), v => syncSource(v), 'image');
    $('#dz').onclick = () => $('#file-input').click();
    $('#dz').addEventListener('dragover', e => { e.preventDefault(); $('#dz').classList.add('over'); });
    $('#dz').addEventListener('dragleave', () => $('#dz').classList.remove('over'));
    $('#dz').addEventListener('drop', e => {
      e.preventDefault(); $('#dz').classList.remove('over');
      if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
    });
    $('#file-input').addEventListener('change', e => { if (e.target.files.length) addFiles(e.target.files); e.target.value = ''; });
    $('#btn-demo').onclick = () => loadDemo(true);
    $('#btn-demo2').onclick = () => loadDemo(true);

    // 视频
    $('#dz-video').onclick = () => $('#file-video').click();
    $('#dz-video').addEventListener('dragover', e => { e.preventDefault(); $('#dz-video').classList.add('over'); });
    $('#dz-video').addEventListener('dragleave', () => $('#dz-video').classList.remove('over'));
    $('#dz-video').addEventListener('drop', e => {
      e.preventDefault(); $('#dz-video').classList.remove('over');
      if (e.dataTransfer.files[0]) submitVideo(e.dataTransfer.files[0]);
    });
    $('#file-video').addEventListener('change', e => { if (e.target.files[0]) submitVideo(e.target.files[0]); e.target.value = ''; });

    // 摄像头
    $('#btn-cam-start').onclick = startCam;
    $('#btn-cam-stop').onclick = stopCam;

    // 队列
    $('#btn-clear-queue').onclick = clearQueue;

    // 参数
    bindRange($('#p-conf'), $('#p-conf-v'), 'params.conf', v => (+v).toFixed(2));
    bindRange($('#p-iou'), $('#p-iou-v'), 'params.iou', v => (+v).toFixed(2));
    bindSeg($('#seg-imgsz'), v => { Store.setPath('params.imgsz', +v, true); BDS.prefs.save(); }, String(S.params.imgsz));
    $('#p-maxdet').addEventListener('change', e => {
      S.params.maxDet = U.clamp(+e.target.value || 300, 1, 3000);
      e.target.value = S.params.maxDet;
      BDS.prefs.save();
    });
    bindSwitch($('#p-slice'), 'params.slice', on => {
      $('#p-slice-opts').hidden = !on;
    });
    setSwitch($('#p-slice'), S.params.slice);
    $('#p-slice-opts').hidden = !S.params.slice;
    // 两个切片滑块的已填充比例：`.rng` 的轨道靠 `--p` 上色，
    // 不更新的话滑块会永远显示默认的 30% —— 拖动后位置与填充对不上。
    const fillOf = el => {
      const min = +el.min, max = +el.max;
      el.style.setProperty('--p', ((+el.value - min) / (max - min) * 100) + '%');
    };
    $('#p-slice-size').addEventListener('input', e => {
      S.params.sliceSize = +e.target.value;
      fillOf(e.target);
      $('#p-slice-v').textContent = S.params.sliceSize + ' · ' + (+S.params.sliceOverlap).toFixed(2);
      BDS.prefs.save();
    });
    $('#p-slice-overlap').addEventListener('input', e => {
      S.params.sliceOverlap = +e.target.value;
      fillOf(e.target);
      $('#p-slice-v').textContent = S.params.sliceSize + ' · ' + (+S.params.sliceOverlap).toFixed(2);
      BDS.prefs.save();
    });
    fillOf($('#p-slice-size')); fillOf($('#p-slice-overlap'));
    $('#p-slice-v').textContent = S.params.sliceSize + ' · ' + (+S.params.sliceOverlap).toFixed(2);

    // 恢复默认推理参数。
    // 这不是"锦上添花"：conf 会持久化，一旦被拖到 0.99，界面就是满屏 0，
    // 而当时的「重置」只清队列/选中，点了没反应 —— 用户只会认为功能坏了。
    $('#btn-params-reset').onclick = () => {
      const d = BDS.DEFAULTS.params;
      S.params.conf = d.conf;
      S.params.iou = d.iou;
      S.params.imgsz = d.imgsz;
      S.params.maxDet = d.maxDet;
      S.params.slice = d.slice;
      S.params.sliceSize = d.sliceSize;
      S.params.sliceOverlap = d.sliceOverlap;
      // 注意：不动 areaM2 —— 那是"这块地多大"的现场标定，不是筛选条件，
      // 顺手清掉会让用户刚填的标定莫名其妙失效。
      BDS.prefs.save();
      syncParamUI();
      renderRight(); renderStage();
      toast('ok', '推理参数已恢复默认',
        '置信度 ' + d.conf + ' · IoU ' + d.iou + ' · 输入 ' + d.imgsz + ' · 最大目标 ' + d.maxDet
        + '（面积标定保持不变）');
    };

    // 操作
    $('#btn-run').onclick = runInference;

    // 人群密度标定：填画面实际覆盖面积 → 重跑当前帧得到密度与分级
    // 回填上次的值，否则输入框空着、但标定其实还生效，看着像没保存
    if (S.params.areaM2 != null && $('#crowd-area')) $('#crowd-area').value = S.params.areaM2;
    renderVideoCrowdTip();
    $('#crowd-apply').onclick = () => {
      const v = parseFloat($('#crowd-area').value);
      S.params.areaM2 = (isFinite(v) && v > 0) ? v : null;
      BDS.prefs.save();
      renderVideoCrowdTip();
      if (S.params.areaM2 == null) {
        toast('warn', '面积无效', '请填入大于 0 的数字（单位 m²），或清空以取消标定');
        renderCrowd(activeFrame());
        return;
      }
      toast('ok', '已标定面积', S.params.areaM2 + ' m² · 重新推理中…');
      runInference();
    };
    $('#btn-reset').onclick = () => {
      clearQueue();
      clearSession();
      S.selected = -1; S.hidden = []; S.search = '';
      $('#det-search').value = '';
      hideBanner();
      setLatency(null); setReqId(null);
      renderRight(); renderStage();
      toast('info', '已重置', '队列、选中状态与会话统计已清空');
    };
    $('#btn-export').onclick = () => exportResults('json');

    // 舞台工具条
    $('#btn-zoom-in').onclick = () => { const s = $('#stage'); Overlay.zoomAt(s.clientWidth / 2, s.clientHeight / 2, 1.2); S.autoFit = false; $('#zoom-val').textContent = Math.round(Overlay.view.scale * 100) + '%'; };
    $('#btn-zoom-out').onclick = () => { const s = $('#stage'); Overlay.zoomAt(s.clientWidth / 2, s.clientHeight / 2, 1 / 1.2); S.autoFit = false; $('#zoom-val').textContent = Math.round(Overlay.view.scale * 100) + '%'; };
    $('#btn-fit').onclick = () => { Overlay.fit(); renderStage(); };
    $('#btn-actual').onclick = () => {
      const s = $('#stage');
      if (!Overlay.img) return;
      Overlay.view.scale = 1;
      Overlay.view.x = (s.clientWidth - Overlay.img.width) / 2;
      Overlay.view.y = (s.clientHeight - Overlay.img.height) / 2;
      Overlay.render(); renderStage();
    };
    $('#btn-full').onclick = () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else $('#stage').requestFullscreen().catch(() => { });
    };
    // 面板开合后画布尺寸会变，需要重算一次标注画布，否则会拉伸变形。
    const afterPanelToggle = () => setTimeout(() => { if (S.view === 'detect') Overlay.resize(); }, 40);

    // 右栏：>1080 收起整列（腾出画布），≤1080 走抽屉
    $('#btn-panel-r').onclick = () => {
      if (window.innerWidth > 1080) $('#view-detect').classList.toggle('no-inspector');
      else $('#col-r').classList.toggle('is-open');
      afterPanelToggle();
    };
    // 左栏：≤900 时是抽屉。工具栏上的 #btn-panel-l 保证收起后仍能重新打开
    // （#btn-hide-l 在面板内部，面板一旦滑出屏幕就点不到了）。
    const toggleLeftPanel = () => { $('#col-l').classList.toggle('is-open'); afterPanelToggle(); };
    $('#btn-panel-l').onclick = toggleLeftPanel;
    $('#btn-hide-l').onclick = toggleLeftPanel;
    $('#btn-hide-r').onclick = () => { $('#col-r').classList.remove('is-open'); afterPanelToggle(); };

    const toggles = [
      ['#tg-boxes', 'boxes'], ['#tg-labels', 'labels'], ['#tg-index', 'index'],
      ['#tg-conf', 'conf'], ['#tg-mask', 'mask'], ['#tg-compare', 'compare'], ['#tg-grid', 'grid'],
    ];
    toggles.forEach(t => {
      const el = $(t[0]);
      el.classList.toggle('is-on', !!S.ui[t[1]]);
      el.onclick = () => {
        S.ui[t[1]] = !S.ui[t[1]];
        el.classList.toggle('is-on', S.ui[t[1]]);
        renderStage();
      };
    });

    // 右栏
    $('#det-search').addEventListener('input', U.debounce(e => { S.search = e.target.value; renderRight(); }, 140));
    $('#det-sort').onclick = () => {
      const order = ['conf', 'confAsc', 'class', 'area'];
      const names = { conf: '置信度 ↓', confAsc: '置信度 ↑', class: '类别', area: '面积 ↓' };
      S.sort = order[(order.indexOf(S.sort) + 1) % order.length];
      $('#det-sort').title = '排序：' + names[S.sort];
      toast('info', '排序方式', names[S.sort], 1600);
      renderRight();
    };
    $('#detail-locate').onclick = () => { if (S.selected >= 0) { Overlay.focusOn(S.selected); renderStage(); } };

    // 任务
    $('#btn-job-refresh').onclick = () => Jobs.refresh();

    // 分析（四段汇总，每段各有自己的刷新入口）
    $('#btn-an-clear').onclick = () => { clearSession(); Analytics.render(); toast('info', '会话统计已清空'); };
    $('#btn-an-export').onclick = () => exportSession();
    $('#btn-an-jobs').onclick = () => Jobs.refresh();
    $('#btn-an-geo').onclick = () => go('geo');
    $('#btn-train-reload').onclick = () => Train.load(true);
    $('#btn-goto-train').onclick = () => go('analytics');

    // 北斗
    $('#btn-geo-copy').onclick = async () => {
      const b = S.beidou || {};
      const ok = !!b.usable && b.lat != null && b.lon != null;
      const parts = [
        ok ? 'LAT ' + Number(b.lat).toFixed(6) : 'LAT —',
        ok ? 'LON ' + Number(b.lon).toFixed(6) : 'LON —',
        'ALT ' + (ok && b.alt != null ? Number(b.alt).toFixed(1) + 'm' : '—'),
        '卫星 ' + (b.satellites_used == null ? '—' : b.satellites_used) +
          '/' + (b.satellites_visible == null ? '—' : b.satellites_visible),
      ];
      if (b.fix_quality_zh) parts.push(b.fix_quality_zh);
      if (b.utc) parts.push('UTC ' + b.utc);
      const txt = parts.join('  ');
      const done = await U.copy(txt);
      toast(done ? 'ok' : 'warn',
        done ? (ok ? '坐标已复制' : '已复制（当前无有效定位）') : '复制失败', txt);
    };

    // 存储卡日志回放（离线路线）
    Track.bind();

    // 设置
    $$('[data-theme-opt]').forEach(el => { el.onclick = () => { applyTheme(el.dataset.themeOpt); Settings.render(); }; });
    $$('[data-pal]').forEach(el => { el.onclick = () => { applyPalette(el.dataset.pal); Settings.render(); }; });
    bindSeg($('#set-density'), v => { applyDensity(v); Settings.render(); }, S.density);
    bindSwitch($('#set-motion'), null, on => applyMotion(on));
    bindSwitch($('#set-autoprobe'), null, on => { S.api.autoprobe = on; BDS.prefs.save(); });
    bindSwitch($('#set-persist'), null, on => { S.persist = on; if (!on) { try { localStorage.removeItem(SESSION_KEY); } catch (e) { } } else saveSession(); });
    bindSwitch($('#set-slice'), null, on => { S.params.slice = on; setSwitch($('#p-slice'), on); $('#p-slice-opts').hidden = !on; BDS.prefs.save(); });
    $('#set-api').addEventListener('change', e => {
      S.api.base = e.target.value.trim();
      BDS.prefs.save();
      testConnection(true);
    });
    $('#set-timeout').addEventListener('change', e => {
      S.api.timeout = U.clamp(+e.target.value || 120, 5, 600);
      e.target.value = S.api.timeout;
      BDS.prefs.save();
    });
    $('#btn-test-conn').onclick = () => testConnection(true);
    $('#set-conf').addEventListener('input', e => {
      S.params.conf = +e.target.value;
      $('#set-conf-v').textContent = (+e.target.value).toFixed(2);
      e.target.style.setProperty('--p', (e.target.value * 100) + '%');
      $('#p-conf').value = e.target.value;
      $('#p-conf-v').textContent = (+e.target.value).toFixed(2);
      $('#p-conf').style.setProperty('--p', (e.target.value * 100) + '%');
      BDS.prefs.save();
      renderRight(); renderStage();
    });
    $('#set-iou').addEventListener('input', e => {
      S.params.iou = +e.target.value;
      $('#set-iou-v').textContent = (+e.target.value).toFixed(2);
      e.target.style.setProperty('--p', (e.target.value * 100) + '%');
      $('#p-iou').value = e.target.value;
      $('#p-iou-v').textContent = (+e.target.value).toFixed(2);
      $('#p-iou').style.setProperty('--p', (e.target.value * 100) + '%');
      BDS.prefs.save();
    });
    $('#set-imgsz').addEventListener('change', e => {
      S.params.imgsz = +e.target.value;
      $$('#seg-imgsz .seg-btn').forEach(b => b.setAttribute('aria-selected', String(+b.dataset.v === S.params.imgsz)));
      BDS.prefs.save();
    });
    $('#btn-export-session').onclick = () => exportSession();
    $('#btn-clear-data').onclick = () => { clearSession(); Analytics.renderSession(); Settings.render(); toast('info', '会话数据已清空'); };

    // 面板收起（窄屏）
    $('#btn-panel-r').onclick = () => {
      const el = $('#col-r');
      if (window.innerWidth <= 1080) el.classList.toggle('is-open');
      else el.style.display = el.style.display === 'none' ? '' : 'none';
    };
  }

  /* ======================================================== 快捷键 */
  function bindKeys() {
    document.addEventListener('keydown', e => {
      const tag = (e.target.tagName || '').toLowerCase();
      const typing = tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable;

      // 命令面板
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if ($('#palette').classList.contains('is-open')) closePalette(); else openPalette();
        return;
      }

      if (e.key === 'Escape') {
        if ($('#palette').classList.contains('is-open')) { closePalette(); return; }
        if ($$('.modal.is-open').length) { closeAllModals(); return; }
        // 窄屏抽屉：Esc 依次关掉右栏、左栏
        if ($('#col-r').classList.contains('is-open')) { $('#col-r').classList.remove('is-open'); return; }
        if ($('#col-l').classList.contains('is-open')) { $('#col-l').classList.remove('is-open'); return; }
        if (S.selected >= 0) { S.selected = -1; renderRight(); renderStage(); return; }
      }

      if (typing) return;

      // Alt + 1..5 切视图
      if (e.altKey && /^[1-5]$/.test(e.key)) {
        e.preventDefault();
        go(VIEWS[+e.key - 1]);
        return;
      }

      const k = e.key;
      const lower = k.toLowerCase();

      if (k === '?' || (e.shiftKey && k === '/')) { e.preventDefault(); openModal('modal-keys'); return; }
      if (lower === 'd' && !e.shiftKey) { applyTheme(S.theme === 'dark' ? 'light' : 'dark'); return; }
      if (e.shiftKey && lower === 'd') { e.preventDefault(); go('detect'); loadDemo(true); return; }

      if (S.view !== 'detect') return;

      if (lower === 'r') { e.preventDefault(); runInference(); return; }
      if (lower === 'o') { e.preventDefault(); $('#file-input').click(); return; }
      if (k === '1' && !e.shiftKey) { syncSource('image'); return; }
      if (k === '2') { syncSource('video'); return; }
      if (k === '3') { syncSource('cam'); return; }
      if (k === '!' || (e.shiftKey && k === '1')) { $('#btn-actual').click(); return; }
      if (lower === 'f') { Overlay.fit(); renderStage(); return; }
      if (k === '+' || k === '=') { $('#btn-zoom-in').click(); return; }
      if (k === '-' || k === '_') { $('#btn-zoom-out').click(); return; }
      if (lower === 'b') { $('#tg-boxes').click(); return; }
      if (lower === 'l') { $('#tg-labels').click(); return; }
      if (lower === 'n') { $('#tg-index').click(); return; }
      if (lower === 'c') { $('#tg-conf').click(); return; }
      if (lower === 'm') { $('#tg-mask').click(); return; }
      if (lower === 'g') { $('#tg-grid').click(); return; }
      if (lower === 'v') { $('#tg-compare').click(); return; }
      if (k === 'ArrowDown' || k === 'ArrowUp') {
        const f = activeFrame();
        if (!f || !f.dets.length) return;
        e.preventDefault();
        const list = filteredDets();
        if (!list.length) return;
        let cur = list.findIndex(d => d._i === S.selected);
        cur = cur < 0 ? -1 : cur;
        const next = U.clamp(cur + (k === 'ArrowDown' ? 1 : -1), 0, list.length - 1);
        S.selected = list[next]._i;
        renderRight(); renderStage();
        const row = $('#det-list .det-row[data-i="' + S.selected + '"]');
        if (row) row.scrollIntoView({ block: 'nearest' });
      }
    });
  }

  /* ======================================================== 初始化 */
  async function init() {
    BDS.prefs.load();
    loadSession();

    document.documentElement.setAttribute('data-theme', S.theme);
    document.documentElement.setAttribute('data-pal', S.palette);
    document.documentElement.setAttribute('data-density', S.density);
    document.documentElement.setAttribute('data-motion', S.motion ? 'on' : 'off');
    $('#btn-theme').innerHTML = '<svg><use href="#' + (S.theme === 'dark' ? 'i-moon' : 'i-sun') + '"/></svg>';

    // 修饰键显示
    if (isMac) {
      U.$$('.cmd-trigger .k').forEach(el => { el.innerHTML = '<kbd>⌘</kbd><kbd>K</kbd>'; });
      U.$$('.statusbar kbd').forEach(el => { if (el.textContent === 'Ctrl') el.textContent = '⌘'; });
    }

    Overlay.init($('#cv'), $('#stage'));
    Overlay.opts = { visible: CLASSES.map(c => c.id), selected: -1, hover: -1 };

    bindUI();
    bindStage();
    bindKeys();
    syncSource('image');

    // 参数回填：统一走 syncParamUI()，避免"启动时一套、重置时另一套"两份逻辑漂移
    syncParamUI();

    renderQueue(); renderRight(); renderFilmstrip(); renderStatus(); clearStage();
    Geo.renderTiles();

    let rt = null;
    window.addEventListener('resize', () => {
      clearTimeout(rt);
      rt = setTimeout(() => {
        // 跨断点时清掉会"卡住"的抽屉状态：宽屏下抽屉必须是打开的，
        // 否则面板会停在屏幕外，而且没有入口能把它拉回来。
        if (window.innerWidth > 900) $('#col-l').classList.remove('is-open');
        if (window.innerWidth > 1080) $('#col-r').classList.remove('is-open');
        if (S.view !== 'detect') return;
        Overlay.resize();
        if (S.autoFit && Overlay.img) Overlay.fit();
        renderStage();
      }, 120);
    });

    // 探测后端
    const online = await testConnection(false);
    if (online) {
      const s = await Api.status();
      if (s.ok) applyStatus(s);
      toast('ok', '后端已连接', '模式：' + (S.model.mode === 'real' ? '真实推理' : '演示模式') +
        (S.model.device ? ' · 设备 ' + (S.model.device === '0' ? 'GPU' : S.model.device) : ''));
    } else {
      showBanner('warn', '未检测到后端服务。界面以离线演示形态运行；启动后端后刷新页面即可获得真实推理。');
    }
    renderStatus();

    // 后端可能已经导入过日志（上次会话留下的），拉一次状态
    if (online) Track.load();

    // 连接探测是**异步**的：用户完全可能在它返回之前就点了「分析」。
    // 那次 go('analytics') 里 S.api.online 还是 false，任务汇总那一次取数被跳过，
    // 而之后没有任何东西会再补一次 —— 页面就一直显示「尚无任务 / 0 已完成」，
    // 后端明明有 8 条任务。所以探测回来是 online 就按当前视图补一次取数。
    // （实测：模型加载要几十秒时最容易撞上，因为这段时间页面已经可交互了。）
    if (online && S.view === 'analytics') Jobs.refresh();
    if (online && S.view === 'geo') Geo.render();

    // 模型选择器：拉取候选权重并绑定切换。
    // 注意：此处不再自动载入演示场景——打开界面即为空态，
    // 需要演示图请手动点击「载入演示场景」。
    const msEl = $('#model-select');
    if (msEl) msEl.onchange = (e) => ModelSelector.switchTo(e.target.value);
    await ModelSelector.refresh();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();

})(window);

/* ============================================================================
 * core.js — 基础层
 *   · util     通用工具
 *   · CLASSES  类别定义
 *   · PALETTES 配色方案
 *   · Store    全局状态（轻量发布订阅）
 *   · Api      接口适配层（自动兼容 FastAPI /api/v1 与旧版 Flask /api）
 *   · Demo     程序化生成的航拍演示场景
 * ========================================================================== */
(function (global) {
  'use strict';

  const BDS = global.BDS || (global.BDS = {});

  /* ----------------------------------------------------------- 通用工具 */
  const U = {
    $: (s, r) => (r || document).querySelector(s),
    $$: (s, r) => Array.prototype.slice.call((r || document).querySelectorAll(s)),

    clamp(v, a, b) { return v < a ? a : v > b ? b : v; },
    lerp(a, b, t) { return a + (b - a) * t; },
    round(v, n) { const p = Math.pow(10, n == null ? 2 : n); return Math.round(v * p) / p; },

    uid() { return Math.random().toString(36).slice(2, 9) + Date.now().toString(36).slice(-3); },

    /** 稳定的伪随机数发生器（mulberry32），用于生成可复现的演示场景 */
    rng(seed) {
      let a = seed >>> 0;
      return function () {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    },

    fmtBytes(n) {
      if (n == null || isNaN(n)) return '—';
      if (n < 1024) return n + ' B';
      if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
      if (n < 1073741824) return (n / 1048576).toFixed(1) + ' MB';
      return (n / 1073741824).toFixed(2) + ' GB';
    },

    fmtNum(n, d) {
      if (n == null || isNaN(n)) return '—';
      return Number(n).toFixed(d == null ? 0 : d).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    },

    fmtMs(ms) {
      if (ms == null || isNaN(ms)) return '—';
      if (ms < 1000) return Math.round(ms) + ' ms';
      return (ms / 1000).toFixed(2) + ' s';
    },

    fmtTime(ts) {
      const d = ts instanceof Date ? ts : new Date(ts);
      if (isNaN(d.getTime())) return '—';
      const p = n => String(n).padStart(2, '0');
      return p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
    },

    fmtDate(ts) {
      const d = ts instanceof Date ? ts : new Date(ts);
      if (isNaN(d.getTime())) return '—';
      const p = n => String(n).padStart(2, '0');
      return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate()) + ' ' +
        p(d.getHours()) + ':' + p(d.getMinutes());
    },

    /** 相对时间：刚刚 / 3 分钟前 / 2 小时前 */
    ago(ts) {
      const s = (Date.now() - new Date(ts).getTime()) / 1000;
      if (isNaN(s)) return '—';
      if (s < 10) return '刚刚';
      if (s < 60) return Math.floor(s) + ' 秒前';
      if (s < 3600) return Math.floor(s / 60) + ' 分钟前';
      if (s < 86400) return Math.floor(s / 3600) + ' 小时前';
      return Math.floor(s / 86400) + ' 天前';
    },

    esc(s) {
      return String(s == null ? '' : s).replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
      ));
    },

    debounce(fn, wait) {
      let t = null;
      return function () {
        const args = arguments, self = this;
        clearTimeout(t);
        t = setTimeout(() => fn.apply(self, args), wait);
      };
    },

    throttle(fn, wait) {
      let last = 0, timer = null;
      return function () {
        const args = arguments, self = this, now = Date.now();
        if (now - last >= wait) { last = now; fn.apply(self, args); }
        else if (!timer) {
          timer = setTimeout(() => { timer = null; last = Date.now(); fn.apply(self, args); }, wait - (now - last));
        }
      };
    },

    /** 极简模糊匹配：支持乱序字符命中，返回得分（0 表示不匹配） */
    fuzzy(q, text) {
      if (!q) return 1;
      const s = q.toLowerCase().trim();
      const t = String(text).toLowerCase();
      if (t.indexOf(s) >= 0) return 100 - t.indexOf(s);
      let i = 0, score = 0, streak = 0;
      for (let j = 0; j < t.length && i < s.length; j++) {
        if (t[j] === s[i]) { i++; streak++; score += 2 + streak; }
        else streak = 0;
      }
      return i === s.length ? score : 0;
    },

    download(name, content, mime) {
      const blob = content instanceof Blob ? content
        : new Blob([content], { type: mime || 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = name;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
    },

    async copy(text) {
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(text); return true;
        }
      } catch (e) { /* 回退 */ }
      try {
        const ta = document.createElement('textarea');
        ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        const ok = document.execCommand('copy'); ta.remove();
        return ok;
      } catch (e) { return false; }
    },

    fileToDataUrl(file) {
      return new Promise((res, rej) => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.onerror = rej;
        fr.readAsDataURL(file);
      });
    },

    loadImage(src) {
      return new Promise((res, rej) => {
        const im = new Image();
        im.onload = () => res(im);
        im.onerror = rej;
        im.src = src;
      });
    },

    /** 判断 Windows / macOS 以显示正确的修饰键 */
    isMac() { return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent); },

    /** 目标尺度分级（COCO 口径：<32² 小目标，<96² 中目标） */
    sizeBucket(w, h) {
      const a = w * h;
      if (a < 32 * 32) return 0;
      if (a < 96 * 96) return 1;
      return 2;
    },
  };
  BDS.util = U;

  /* ----------------------------------------------------------- 类别 */
  BDS.CLASSES = [
    { id: 0, zh: '行人', en: 'pedestrian' },
    { id: 1, zh: '人群', en: 'people' },
    { id: 2, zh: '自行车', en: 'bicycle' },
    { id: 3, zh: '小汽车', en: 'car' },
    { id: 4, zh: '厢式货车', en: 'van' },
    { id: 5, zh: '卡车', en: 'truck' },
    { id: 6, zh: '三轮车', en: 'tricycle' },
    { id: 7, zh: '带棚三轮车', en: 'awning-tricycle' },
    { id: 8, zh: '公交车', en: 'bus' },
    { id: 9, zh: '摩托车', en: 'motor' },
  ];

  BDS.PALETTES = {
    soft: {
      dark: ['#4ade80', '#f87171', '#38bdf8', '#fbbf24', '#a78bfa', '#fb923c', '#94a3b8', '#f472b6', '#2dd4bf', '#818cf8'],
      light: ['#16a34a', '#dc2626', '#0284c7', '#ca8a04', '#7c3aed', '#ea580c', '#64748b', '#db2777', '#0d9488', '#4f46e5'],
    },
    vivid: {
      dark: ['#22c55e', '#ef4444', '#06b6d4', '#eab308', '#8b5cf6', '#f97316', '#64748b', '#ec4899', '#14b8a6', '#6366f1'],
      light: ['#15803d', '#b91c1c', '#0e7490', '#a16207', '#6d28d9', '#c2410c', '#475569', '#be185d', '#0f766e', '#4338ca'],
    },
    mono: {
      dark: ['#93c5fd', '#a5c9ff', '#7ea6f0', '#c0d8ff', '#8ab4f8', '#b8d4ff', '#6f9ae8', '#dbe7ff', '#9ec1ff', '#cfe2ff'],
      light: ['#1e40af', '#1d4ed8', '#2563eb', '#1e3a8a', '#1d4ed8', '#3b82f6', '#1e40af', '#60a5fa', '#2563eb', '#1e3a8a'],
    },
  };

  /* ----------------------------------------------------------- 状态 */
  const DEFAULTS = {
    view: 'detect',
    theme: 'dark',
    palette: 'soft',
    density: 'comfortable',
    motion: true,

    api: {
      base: '',
      autoprobe: true,
      timeout: 120,
      online: false,
      prefix: '/api/v1',
      shape: 'fastapi',      // fastapi | flask | none
      version: null,
      lastProbe: 0,
    },

    model: {
      mode: 'demo',
      device: null,
      cudaName: null,
      weights: null,
      loaded: false,
      classes: 10,
      engineAvailable: false,
      slicingAvailable: false,
      message: null,
    },

    // 定位帧。字段**全部可空**：没有定位时 lat/lon 就是 null，
    // 界面必须显示"—"，绝不能把它当 0（Number(null) === 0，会渲染成 0.000000，
    // 看起来像"定位在几内亚湾"）。这一点是刻意设计的，别"顺手补默认值"。
    beidou: {
      lat: null, lon: null, alt: null, satellites: null, real: false,
      source: 'unavailable', source_detail: null, usable: false,
      fix_quality: null, fix_quality_zh: null, fix_type: null, fix_type_zh: null,
      mode_zh: null, hdop: null, vdop: null, pdop: null,
      satellites_used: null, satellites_visible: null, satellites_detail: [],
      speed_kmh: null, course: null, utc: null, age_s: null, stale: false,
      sentences: 0, checksum_errors: 0, message: null,
      gsd_m_per_px: 0, gsd_calibrated: false, estimated: false,
      // 北斗特色服务三块。**初值一律 null / 空对象**，不预置任何"看起来正常"
      // 的默认档位或时间 —— 后端没给就是没给，界面显示 "—" 并说明原因。
      accuracy_tier: null, accuracy_message: null,
      timebase: null, composition: null,
    },

    // areaM2：画面实际覆盖面积（m²），人群密度标定用；null = 未标定，后端只给人数不给密度
    params: { conf: 0.25, iou: 0.45, imgsz: 640, maxDet: 300, slice: false, sliceSize: 640, sliceOverlap: 0.2, areaM2: null },

    // 存储卡日志轨迹（离线路线）。loaded=false 时其余字段无意义，界面显示空状态。
    track: {
      loaded: false, name: null, error: null,
      summary: null, diagnose: null, polyline: [],
      // 本地插值序列：一次性下发，拖动时间轴时**不再发任何请求**。
      // 逐帧问后端会把浏览器连接池拖死（实测 seek 卡满 120 s 客户端超时）。
      scrub: [],          // [{t, lat, lon, alt, ..., sat}]，t 是**日志时间轴**上的秒
      scrubStride: 1,     // 抽稀步长（>1 说明时间轴上的读数来自插值）
      scrubSats: [],      // 卫星明细去重表，点的 sat 字段是它的下标
      t: 0,               // 当前拖到的时间轴位置（视频秒）
      fix: null,          // 该时刻的定位帧（本地插值得到）
      loading: false,
    },

    queue: [],       // {id,name,size,type,url,thumb}
    activeId: null,
    frames: {},      // id -> {img,width,height,dets,ms,engine,device,src}

    ui: { boxes: true, labels: true, index: false, conf: true, mask: false, compare: false, grid: false },
    zoom: 1, panX: 0, panY: 0, autoFit: true,
    selected: -1, hover: -1,
    hidden: [],      // 被隐藏的类别 id
    sort: 'conf',    // conf | confAsc | class | area
    search: '',

    session: {
      frames: 0, dets: 0, sumConf: 0, confCount: 0,
      latencies: [], classCount: new Array(10).fill(0),
      confHist: new Array(20).fill(0), history: [],
      smallDets: 0, allDets: 0,
    },

    jobs: [],
    camOn: false,
    source: 'image',
    running: false,
    persist: true,
  };

  function deepClone(o) { return JSON.parse(JSON.stringify(o)); }

  const Store = {
    state: deepClone(DEFAULTS),
    _subs: [],

    get() { return this.state; },

    set(patch, silent) {
      Object.assign(this.state, patch);
      if (!silent) this.emit();
    },

    /** 深路径赋值：set('params.conf', 0.5) */
    setPath(path, value, silent) {
      const ks = path.split('.');
      let o = this.state;
      for (let i = 0; i < ks.length - 1; i++) o = o[ks[i]];
      o[ks[ks.length - 1]] = value;
      if (!silent) this.emit();
    },

    getPath(path) {
      return path.split('.').reduce((o, k) => (o == null ? o : o[k]), this.state);
    },

    subscribe(fn) { this._subs.push(fn); return () => { this._subs = this._subs.filter(f => f !== fn); }; },

    emit() { for (const f of this._subs.slice()) { try { f(this.state); } catch (e) { console.error(e); } } },

    reset() { this.state = deepClone(DEFAULTS); this.emit(); },
  };
  BDS.Store = Store;
  BDS.DEFAULTS = DEFAULTS;

  /** 颜色查询：按当前主题与配色方案返回 10 类颜色 */
  BDS.classColor = function (id) {
    const pal = BDS.PALETTES[Store.state.palette] || BDS.PALETTES.soft;
    const set = pal[Store.state.theme] || pal.dark;
    return set[id % set.length] || '#4d9eff';
  };
  BDS.classColors = function () {
    return BDS.CLASSES.map(c => BDS.classColor(c.id));
  };

  /* ----------------------------------------------------------- 本地持久化 */
  BDS.prefs = {
    KEY: 'bds.prefs.v1',
    save() {
      try {
        const s = Store.state;
        localStorage.setItem(this.KEY, JSON.stringify({
          theme: s.theme, palette: s.palette, density: s.density, motion: s.motion,
          api: { base: s.api.base, autoprobe: s.api.autoprobe, timeout: s.api.timeout },
          params: s.params,
        }));
      } catch (e) { /* file:// 下可能不可用，忽略 */ }
    },
    load() {
      try {
        const raw = localStorage.getItem(this.KEY);
        if (!raw) return;
        const p = JSON.parse(raw);
        const s = Store.state;
        if (p.theme) s.theme = p.theme;
        if (p.palette) s.palette = p.palette;
        if (p.density) s.density = p.density;
        if (typeof p.motion === 'boolean') s.motion = p.motion;
        if (p.api) Object.assign(s.api, p.api);
        if (p.params) Object.assign(s.params, p.params);
      } catch (e) { /* ignore */ }
    },
  };

  /* ----------------------------------------------------------- 接口适配层 */
  const Api = {
    /** 候选路径表：新 FastAPI（/api/v1/*）优先，回退旧 Flask（/api/*） */
    PATHS: {
      health: ['/api/v1/health', '/api/health', '/health'],
      status: ['/api/v1/status', '/api/status', '/status'],
      classes: ['/api/v1/classes', '/api/classes'],
      beidou: ['/api/v1/beidou', '/api/beidou'],
      detect: ['/api/v1/detect', '/api/detect'],
      // JSON(base64) 是**独立端点**：FastAPI 的 /detect 只吃 multipart，
      // 把 JSON 打过去会 400（Form 字段不会被解析）。
      detectJson: ['/api/v1/detect/json', '/api/detect'],
      // 后端只有单数 POST /api/v1/video；把真实路径放最前，
      // 避免被 / 挂载的 StaticFiles 兜底成「405 Method Not Allowed」。
      // ⚠ 这组**只用于 POST 建任务**。列表查询用的是 videoList（见下），
      // 别把两者混在一张表里 —— 那正是「任务汇总永远是空的」的原因。
      videos: ['/api/v1/video', '/api/v1/videos', '/api/video'],
      // 列出近期任务：后端是 GET /api/v1/jobs（复数、且不叫 video）。
      // 只列真实路径：候选表里塞猜测项会白白多打几次 404。
      videoList: ['/api/v1/jobs'],
      // 后端实际路径是 /api/v1/jobs/{id}（见 CreateVideoResponse.status_url）。
      // 写 /api/v1/video/{id} 会被挂载在 "/" 的 StaticFiles 兜底，
      // GET 直接返回 index.html（200 + HTML），比 404 更难排查。
      videoJob: ['/api/v1/jobs/{id}', '/api/v1/job/{id}'],
      files: ['/api/v1/files/{name}', '/api/file/{name}'],
      metrics: ['/api/v1/metrics', '/api/metrics'],
      models: ['/api/v1/models', '/api/models'],
      modelSwitch: ['/api/v1/models/switch', '/api/models/switch'],
      // 存储卡日志轨迹（离线路线）。后端只有这几个真实路径，不要加猜测项 ——
      // 挂载在 "/" 的 StaticFiles 会把不存在的 POST 兜底成 405，
      // 而业务错误处理若把它当致命失败，就会在正确路径之前先弹一个错。
      track: ['/api/v1/track'],
      trackImport: ['/api/v1/track/import'],
      trackAt: ['/api/v1/track/at'],
      trackAlign: ['/api/v1/track/align'],
    },

    url(path) {
      const base = (Store.state.api.base || '').replace(/\/+$/, '');
      return base + path;
    },

    /** 统一请求；返回 {ok,status,data,requestId,ms,error} */
    async request(path, opts) {
      opts = opts || {};
      const timeout = (Store.state.api.timeout || 120) * 1000;
      const ctrl = typeof AbortController !== 'undefined' ? new AbortController() : null;
      const timer = ctrl ? setTimeout(() => ctrl.abort(), timeout) : null;
      const t0 = performance.now();
      try {
        const res = await fetch(this.url(path), Object.assign({}, opts, ctrl ? { signal: ctrl.signal } : {}));
        const ms = performance.now() - t0;
        const rid = res.headers.get('x-request-id') || res.headers.get('X-Request-ID') || null;
        let data = null;
        const ct = res.headers.get('content-type') || '';
        if (ct.indexOf('application/json') >= 0) {
          try { data = await res.json(); } catch (e) { data = null; }
        } else {
          try { data = await res.text(); } catch (e) { data = null; }
        }
        if (!res.ok) {
          const env = data && data.error ? data.error : null;
          return {
            ok: false, status: res.status, data, requestId: rid || (data && data.request_id) || null, ms,
            error: (env && env.message) || ('HTTP ' + res.status),
            code: (env && env.code) || ('HTTP_' + res.status),
          };
        }
        return { ok: true, status: res.status, data, requestId: rid || (data && data.request_id) || null, ms };
      } catch (e) {
        return {
          ok: false, status: 0, data: null, requestId: null, ms: performance.now() - t0,
          error: e && e.name === 'AbortError' ? '请求超时（' + (Store.state.api.timeout) + 's）' : ('网络不可达：' + (e && e.message ? e.message : e)),
          code: 'NETWORK_ERROR',
        };
      } finally {
        if (timer) clearTimeout(timer);
      }
    },

    async get(path) { return this.request(path, { method: 'GET' }); },

    /** 探测后端：依次尝试候选路径，命中即锁定 prefix 与 shape */
    async probe() {
      const s = Store.state;
      if (!s.api.autoprobe && !s.api.base) {
        s.api.online = false; s.api.shape = 'none';
        return { online: false };
      }
      for (const p of this.PATHS.status) {
        const r = await this.request(p, { method: 'GET' });
        if (r.ok && r.data) {
          s.api.online = true;
          s.api.prefix = p.indexOf('/api/v1') === 0 ? '/api/v1' : (p.indexOf('/api/') === 0 ? '/api' : '');
          s.api.shape = p.indexOf('/api/v1') === 0 ? 'fastapi' : 'flask';
          s.api.lastProbe = Date.now();
          return { online: true, data: r.data, shape: s.api.shape, prefix: s.api.prefix, requestId: r.requestId };
        }
      }
      s.api.online = false;
      s.api.shape = 'none';
      return { online: false };
    },

    /** 取 status 并按两种后端形态归一化 */
    async status() {
      const paths = Store.state.api.prefix
        ? [Store.state.api.prefix + '/status'].concat(this.PATHS.status)
        : this.PATHS.status;
      for (const p of paths) {
        const r = await this.request(p, { method: 'GET' });
        if (r.ok && r.data) return { ok: true, data: this.normalizeStatus(r.data), raw: r.data, requestId: r.requestId };
      }
      return { ok: false };
    },

    normalizeStatus(d) {
      const classes = Array.isArray(d.classes) && d.classes.length
        ? d.classes.map(c => ({ id: c.id, zh: c.name_zh || c.zh, en: c.name_en || c.en, color: c.color }))
        : (d.classes_zh || []).map((zh, i) => ({
            id: i, zh: zh, en: (d.classes_en || [])[i] || '', color: (d.class_colors || [])[i] || null,
          }));
      const m = d.model || {};
      return {
        mode: d.mode === 'real' ? 'real' : 'demo',
        device: d.device != null ? String(d.device) : null,
        cudaName: m.cuda_name || null,
        weights: m.weights || null,
        loaded: !!m.loaded || d.mode === 'real',
        classes: classes.length ? classes : BDS.CLASSES.map(c => ({ id: c.id, zh: c.zh, en: c.en, color: null })),
        beidou: d.beidou || null,
        engineAvailable: d.engine_available !== undefined ? !!d.engine_available : (d.mode === 'real'),
        slicingAvailable: d.slicing_available !== undefined ? !!d.slicing_available : (d.mode === 'real'),
        message: d.message || m.reason || null,
      };
    },

    /* ---------------------------------------------------- 存储卡日志轨迹 */

    /** 导入日志。返回后端回执（含诊断），失败时也带回执以便展示原因。 */
    async trackImport(file) {
      const fd = new FormData();
      fd.append('file', file, file.name || 'card.log');
      const r = await this.request(this.PATHS.trackImport[0], { method: 'POST', body: fd });
      if (!r.ok) return { ok: false, error: r.error, data: r.data };
      return { ok: true, data: r.data };
    },

    async trackState() {
      const r = await this.request(this.PATHS.track[0], { method: 'GET' });
      return r.ok ? { ok: true, data: r.data } : { ok: false, error: r.error };
    },

    /**
     * 按视频时刻取单帧。**不要用在拖动/回放的循环里。**
     *
     * 界面上的取帧一律走 ``Track.frameAt()`` 的本地插值 —— 这条接口每次调用
     * 都是一个 HTTP 请求，一秒 60 次会把浏览器的连接池拖死（实测表现为请求
     * 全部挂到 120 s 客户端超时、页面看起来卡住）。留着它是给探针和
     * 单次查询用的。
     */
    async trackAt(t, interpolate) {
      const q = '?t=' + encodeURIComponent(t) +
        (interpolate === false ? '&interpolate=false' : '');
      const r = await this.request(this.PATHS.trackAt[0] + q, { method: 'GET' });
      return r.ok ? { ok: true, data: r.data } : { ok: false, error: r.error };
    },

    async trackAlign(payload) {
      const r = await this.request(this.PATHS.trackAlign[0], {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload || {}),
      });
      return r.ok ? { ok: true, data: r.data } : { ok: false, error: r.error, data: r.data };
    },

    async trackClear() {
      const r = await this.request(this.PATHS.track[0], { method: 'DELETE' });
      return r.ok ? { ok: true } : { ok: false, error: r.error };
    },

    /** 图像检测：文件走 multipart，dataURL 走 JSON（两者是不同端点） */    async detect(input, params, onProgress) {
      const s = Store.state;
      const isFile = input instanceof Blob || input instanceof File;
      const cand = isFile ? this.PATHS.detect : this.PATHS.detectJson;
      const paths = s.api.prefix
        ? [s.api.prefix + (isFile ? '/detect' : '/detect/json')].concat(cand)
        : cand;
      const p = Object.assign({}, s.params, params || {});
      let last = null;

      for (const path of paths) {
        let opts;
        if (isFile) {
          const fd = new FormData();
          fd.append('file', input, input.name || 'image.jpg');
          fd.append('conf', String(0.05));
          fd.append('iou', String(p.iou));
          fd.append('imgsz', String(p.imgsz));
          fd.append('max_det', String(p.maxDet));
          if (p.slice) { fd.append('slice', 'true'); fd.append('slice_size', String(p.sliceSize)); fd.append('slice_overlap', String(p.sliceOverlap)); }
          if (p.areaM2 != null && p.areaM2 > 0) fd.append('area_m2', String(p.areaM2));
          opts = { method: 'POST', body: fd };
        } else {
          opts = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image: input, conf: 0.05, iou: p.iou, imgsz: p.imgsz, max_det: p.maxDet,
              slice: !!p.slice, slice_size: p.sliceSize, slice_overlap: p.sliceOverlap,
              area_m2: (p.areaM2 != null && p.areaM2 > 0) ? p.areaM2 : null,
            }),
          };
        }
        if (onProgress) onProgress(path);
        const r = await this.request(path, opts);
        if (r.ok && r.data) return { ok: true, data: this.normalizeDetect(r.data), requestId: r.requestId, ms: r.ms };
        last = r;
        if (r.status === 404) continue;   // 该路径不存在，试下一个
        return { ok: false, error: r.error, code: r.code, requestId: r.requestId };
      }
      return { ok: false, error: (last && last.error) || '检测接口不可用', code: 'NO_ENDPOINT' };
    },

    normalizeDetect(d) {
      const dets = (d.detections || []).map(x => {
        const b = x.bbox || x;
        return {
          cls: x.class_id != null ? x.class_id : (x.class != null ? x.class : 0),
          conf: +x.conf,
          x1: +b.x1, y1: +b.y1, x2: +b.x2, y2: +b.y2,
          lon: x.lon != null ? +x.lon : null,
          lat: x.lat != null ? +x.lat : null,
        };
      }).filter(x => isFinite(x.conf) && isFinite(x.x1));
      return {
        mode: d.mode || 'real',
        width: d.width || 0,
        height: d.height || 0,
        detections: dets,
        // 人群密度分析（未标定后端也会给人数，density 为 null）
        crowd: d.crowd || null,
        original: d.original || null,
        annotated: d.annotated || null,
        device: d.device != null ? String(d.device) : null,
        engine: d.engine || null,
        beidou: d.beidou || null,
        message: d.message || null,
        elapsedMs: d.elapsed_ms != null ? +d.elapsed_ms : null,
      };
    },

    /**
     * 训练指标（真实训练产物，只读）。
     * 后端解析已加载权重所属 run 的 results.csv / args.yaml / 对比表后返回。
     */
    async metrics() {
      const s = Store.state;
      const paths = s.api.prefix ? [s.api.prefix + '/metrics'].concat(this.PATHS.metrics) : this.PATHS.metrics;
      for (const p of paths) {
        const r = await this.request(p, { method: 'GET' });
        if (r.ok && r.data && typeof r.data.available !== 'undefined') {
          return { ok: true, data: r.data, requestId: r.requestId };
        }
      }
      return { ok: false };
    },

    /** 训练配图的绝对地址（后端直出图片，不占用 dataURL） */
    metricsArtifact(name) {
      const prefix = Store.state.api.prefix || '/api/v1';
      return this.url(prefix + '/metrics/artifact/' + encodeURIComponent(name));
    },

    /** 视频：新后端为异步任务，旧后端为同步返回 */
    async submitVideo(file, params) {
      const s = Store.state;
      const p = Object.assign({}, s.params, params || {});
      const paths = s.api.prefix ? [s.api.prefix + '/video'].concat(this.PATHS.videos) : this.PATHS.videos;
      let last = null;
      for (const path of paths) {
        const fd = new FormData();
        fd.append('file', file, file.name || 'video.mp4');
        fd.append('conf', String(p.conf));
        fd.append('iou', String(p.iou));
        fd.append('imgsz', String(p.imgsz));
        // 画面实际覆盖面积（m²）：后端据此把逐帧人数换算成密度并分级。
        // 不传也能跑，只是只有人数、没有密度与预警。
        if (p.areaM2 != null && p.areaM2 > 0) fd.append('area_m2', String(p.areaM2));
        const r = await this.request(path, { method: 'POST', body: fd });
        if (r.ok && r.data) {
          const d = r.data;
          if (d.job_id && d.status) {
            return { ok: true, async: true, jobId: d.job_id, status: d.status, statusUrl: d.status_url, message: d.message };
          }
          if (d.video_url || d.stats) {
            return { ok: true, async: false, videoUrl: d.video_url, stats: d.stats, jobId: d.job || null };
          }
          return { ok: true, async: false, videoUrl: null, stats: null, raw: d };
        }
        last = r;
        // 路径不存在（404）或方法不允许（405，后端没有该路由时被 StaticFiles 兜底）
        // 都继续尝试下一个候选；其它错误（如 400/413）才直接返回。
        if (r.status === 404 || r.status === 405) continue;
        return { ok: false, error: r.error, code: r.code };
      }
      return { ok: false, error: (last && last.error) || '视频接口不可用', code: 'NO_ENDPOINT' };
    },

    /** 查询视频任务状态 */
    async videoJob(jobId, statusUrl) {
      const urls = [];
      if (statusUrl) urls.push(statusUrl);
      for (const t of this.PATHS.videoJob) urls.push(t.replace('{id}', jobId));
      for (const u of urls) {
        const r = await this.request(u, { method: 'GET' });
        if (r.ok && r.data && (r.data.status || r.data.job_id)) return { ok: true, data: r.data };
      }
      return { ok: false };
    },

    async videoJobs() {
      const s = Store.state;
      const paths = s.api.prefix ? [s.api.prefix + '/jobs'].concat(this.PATHS.videoList) : this.PATHS.videoList;
      for (const p of paths) {
        const r = await this.request(p, { method: 'GET' });
        if (r.ok && Array.isArray(r.data)) return { ok: true, data: r.data };
        if (r.ok && r.data && Array.isArray(r.data.items)) return { ok: true, data: r.data.items };
      }
      return { ok: false };
    },

    /** 拼接后端返回的相对资源地址 */
    asset(u) {
      if (!u) return null;
      if (/^(https?:|data:|blob:)/.test(u)) return u;
      return this.url(u.charAt(0) === '/' ? u : '/' + u);
    },

    /** 拉取权重候选与当前加载情况 */
    async models() {
      const s = Store.state;
      const paths = s.api.prefix ? [s.api.prefix + '/models'].concat(this.PATHS.models) : this.PATHS.models;
      for (const p of paths) {
        const r = await this.request(p, { method: 'GET' });
        if (r.ok && r.data) return { ok: true, data: r.data };
        if (r.status !== 404) return { ok: false, error: r.error };
      }
      return { ok: false, error: '接口未找到' };
    },

    /** 切换当前加载的权重（进程内原地替换） */
    async switchModel(weight) {
      const s = Store.state;
      const paths = s.api.prefix ? [s.api.prefix + '/models/switch'].concat(this.PATHS.modelSwitch) : this.PATHS.modelSwitch;
      for (const p of paths) {
        const r = await this.request(p, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ weight: weight }),
        });
        if (r.ok) return { ok: true, data: r.data };
        // 400 表示后端拒绝（不在候选列表/文件缺失/加载失败），直接返回错误
        if (r.status === 400 || r.status === 422) return { ok: false, error: r.error, raw: r.data };
        if (r.status !== 404) return { ok: false, error: r.error };
      }
      return { ok: false, error: '接口未找到' };
    },
  };
  BDS.Api = Api;

  /* ----------------------------------------------------------- 演示场景生成 */
  const Demo = {
    /**
     * 程序化生成一幅"卫星底图"风格的航拍场景，并附带与之严格对应的检测结果。
     * 目的：在没有后端 / 没有权重的情况下，也能完整演示全部交互与可视化。
     */
    build(seed) {
      const W = 1600, H = 1100;
      const rand = U.rng(seed || 20260913);
      const pick = arr => arr[Math.floor(rand() * arr.length)];
      const rint = (a, b) => a + Math.floor(rand() * (b - a + 1));
      const rf = (a, b) => a + rand() * (b - a);

      // 2× 超采样绘制，最后降采样回目标尺寸：边缘更柔和、噪点更细腻，
      // 明显削弱"矢量插画感"，更接近真实航拍成像。
      const SS = 2;
      const cv = document.createElement('canvas');
      cv.width = W * SS; cv.height = H * SS;
      const g = cv.getContext('2d');
      g.scale(SS, SS);

      /* --- 底色：白天航拍的地表（浅混凝土色，才能压出阴影层次） --- */
      g.fillStyle = '#8a8474';
      g.fillRect(0, 0, W, H);

      /* --- 地表纹理 --- */
      for (let i = 0; i < 4200; i++) {
        const x = rand() * W, y = rand() * H, s = rf(1, 4);
        const v = rint(96, 176);
        g.fillStyle = 'rgba(' + v + ',' + (v + rint(2, 14)) + ',' + (v - rint(4, 14)) + ',' + rf(0.06, 0.26).toFixed(2) + ')';
        g.fillRect(x, y, s, s);
      }
      // 大块地表色差，避免"纯色板"
      for (let i = 0; i < 26; i++) {
        g.fillStyle = 'rgba(' + rint(112, 168) + ',' + rint(114, 170) + ',' + rint(100, 156) + ',0.12)';
        const bw = rf(90, 320), bh = rf(90, 300);
        g.beginPath();
        g.ellipse(rand() * W, rand() * H, bw / 2, bh / 2, rf(0, 3.14), 0, Math.PI * 2);
        g.fill();
      }

      /* --- 道路网 --- */
      const roads = [
        { x: 0, y: 420, w: W, h: 150, dir: 'h' },
        { x: 0, y: 800, w: W, h: 100, dir: 'h' },
        { x: 600, y: 0, w: 150, h: H, dir: 'v' },
        { x: 1250, y: 0, w: 120, h: H, dir: 'v' },
      ];
      // 路肩投影
      g.fillStyle = 'rgba(52,54,48,0.42)';
      roads.forEach(r => g.fillRect(r.x - 8, r.y - 8, r.w + 16, r.h + 16));
      // 人行道（浅灰，比地表更亮）
      g.fillStyle = '#b2b2a8';
      roads.forEach(r => g.fillRect(r.x - 7, r.y - 7, r.w + 14, r.h + 14));
      // 沥青路面
      g.fillStyle = '#63666b';
      roads.forEach(r => g.fillRect(r.x, r.y, r.w, r.h));
      // 路面细微磨损
      for (let i = 0; i < 900; i++) {
        const r = roads[Math.floor(rand() * roads.length)];
        const x = r.x + rand() * r.w, y = r.y + rand() * r.h;
        g.fillStyle = 'rgba(' + rint(74, 152) + ',' + rint(74, 152) + ',' + rint(76, 154) + ',0.14)';
        g.fillRect(x, y, rf(2, 9), rf(2, 7));
      }
      // 车道轮迹（比路面略深，增加方向感）
      g.fillStyle = 'rgba(40,42,46,0.16)';
      roads.forEach(r => {
        if (r.dir === 'h') {
          [0.2, 0.42, 0.58, 0.8].forEach(t => g.fillRect(0, r.y + r.h * t, W, 7));
        } else {
          [0.2, 0.42, 0.58, 0.8].forEach(t => g.fillRect(r.x + r.w * t, 0, 7, H));
        }
      });

      // 路缘白线
      g.strokeStyle = 'rgba(232,232,220,0.85)'; g.lineWidth = 2.5;
      roads.forEach(r => {
        if (r.dir === 'h') {
          g.beginPath(); g.moveTo(0, r.y + 3); g.lineTo(W, r.y + 3); g.stroke();
          g.beginPath(); g.moveTo(0, r.y + r.h - 3); g.lineTo(W, r.y + r.h - 3); g.stroke();
        } else {
          g.beginPath(); g.moveTo(r.x + 3, 0); g.lineTo(r.x + 3, H); g.stroke();
          g.beginPath(); g.moveTo(r.x + r.w - 3, 0); g.lineTo(r.x + r.w - 3, H); g.stroke();
        }
      });

      // 中央虚线
      g.strokeStyle = 'rgba(240,240,228,0.78)'; g.lineWidth = 3.5;
      g.setLineDash([30, 32]);
      roads.forEach(r => {
        g.beginPath();
        if (r.dir === 'h') { const y = r.y + r.h / 2; g.moveTo(0, y); g.lineTo(W, y); }
        else { const x = r.x + r.w / 2; g.moveTo(x, 0); g.lineTo(x, H); }
        g.stroke();
      });
      g.setLineDash([]);

      // 斑马线
      function crosswalk(cx, cy, w, h, vertical) {
        g.fillStyle = 'rgba(238,238,228,0.9)';
        if (!vertical) {
          for (let x = cx; x < cx + w; x += 24) g.fillRect(x, cy, 13, h);
        } else {
          for (let y = cy; y < cy + h; y += 24) g.fillRect(cx, y, w, 13);
        }
      }
      crosswalk(556, 420, 44, 150, true);
      crosswalk(750, 420, 44, 150, true);
      crosswalk(600, 376, 150, 44, false);
      crosswalk(1250, 796, 44, 104, true);

      /* --- 街区（建筑 / 绿地 / 水体）--- */
      const blocks = [
        { x: 16, y: 16, w: 568, h: 388 },
        { x: 766, y: 16, w: 468, h: 388 },
        { x: 1386, y: 16, w: 198, h: 388 },
        { x: 16, y: 586, w: 568, h: 198 },
        { x: 766, y: 586, w: 468, h: 198 },
        { x: 1386, y: 586, w: 198, h: 198 },
        { x: 16, y: 916, w: 568, h: 168 },
        { x: 766, y: 916, w: 468, h: 168 },
        { x: 1386, y: 916, w: 198, h: 168 },
      ];

      const buildings = [];

      blocks.forEach((b, bi) => {
        const kind = (bi === 2 || bi === 5) ? 'park' : (bi === 7 ? 'water' : 'urban');
        if (kind === 'park') {
          g.fillStyle = '#557038';
          g.fillRect(b.x, b.y, b.w, b.h);
          g.strokeStyle = '#43592c'; g.lineWidth = 2; g.strokeRect(b.x + 1, b.y + 1, b.w - 2, b.h - 2);
          // 草地纹理
          for (let i = 0; i < 260; i++) {
            g.fillStyle = 'rgba(' + rint(60, 110) + ',' + rint(110, 160) + ',' + rint(50, 90) + ',0.28)';
            g.fillRect(b.x + rand() * b.w, b.y + rand() * b.h, rf(2, 7), rf(2, 6));
          }
          for (let i = 0; i < 18; i++) {
            const tx = b.x + rf(14, b.w - 14), ty = b.y + rf(14, b.h - 14), r = rf(7, 15);
            g.beginPath(); g.arc(tx + 3, ty + 4, r, 0, Math.PI * 2); g.fillStyle = 'rgba(30,44,22,0.55)'; g.fill();
            g.beginPath(); g.arc(tx, ty, r, 0, Math.PI * 2); g.fillStyle = '#3f6330'; g.fill();
            g.beginPath(); g.arc(tx - r * 0.28, ty - r * 0.32, r * 0.62, 0, Math.PI * 2); g.fillStyle = '#4d7a3c'; g.fill();
          }
          return;
        }
        if (kind === 'water') {
          // 不规则岸线：带扰动的闭合曲线，避免"矩形泳池"感
          const seg = 18, inset = 26;
          const pts = [];
          for (let i = 0; i < seg; i++) {
            const t = (i / seg) * Math.PI * 2;
            const rx = Math.max(20, b.w / 2 - inset) * (1 + rf(-0.18, 0.18));
            const ry = Math.max(20, b.h / 2 - inset) * (1 + rf(-0.22, 0.22));
            pts.push([b.x + b.w / 2 + Math.cos(t) * rx, b.y + b.h / 2 + Math.sin(t) * ry]);
          }
          function shorePath() {
            g.beginPath();
            g.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < pts.length; i++) {
              const p = pts[i], q = pts[(i + 1) % pts.length];
              g.quadraticCurveTo(p[0], p[1], (p[0] + q[0]) / 2, (p[1] + q[1]) / 2);
            }
            g.closePath();
          }
          // 岸边绿化带
          shorePath();
          g.save();
          g.lineWidth = 20;
          g.strokeStyle = 'rgba(78,108,60,0.9)';
          g.lineJoin = 'round';
          g.stroke();
          g.restore();

          // 水面 + 波纹
          shorePath();
          g.save();
          g.fillStyle = '#3d6b83';
          g.fill();
          g.clip();
          g.fillStyle = '#4a7d97';
          g.beginPath();
          g.ellipse(b.x + b.w * 0.4, b.y + b.h * 0.38, b.w * 0.34, b.h * 0.3, 0, 0, Math.PI * 2);
          g.fill();
          for (let i = 0; i < 48; i++) {
            g.strokeStyle = 'rgba(226,242,255,' + rf(0.08, 0.26).toFixed(2) + ')';
            g.lineWidth = rf(1, 2);
            const y = b.y + rf(6, b.h - 6);
            g.beginPath(); g.moveTo(b.x + rf(0, b.w * 0.4), y);
            g.lineTo(b.x + rf(b.w * 0.5, b.w - 6), y + rf(-3, 3)); g.stroke();
          }
          g.restore();
          return;
        }

        // 城市街区：切分成若干栋楼
        const cols = b.w > 460 ? 2 : 1;
        const rows = b.h > 300 ? 2 : 1;
        const gap = 14;
        const cw = (b.w - gap * (cols + 1)) / cols;
        const ch = (b.h - gap * (rows + 1)) / rows;
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const bx = b.x + gap + c * (cw + gap);
            const by = b.y + gap + r * (ch + gap);
            const bw = cw * rf(0.72, 1);
            const bh = ch * rf(0.72, 1);
            buildings.push({ x: bx, y: by, w: bw, h: bh });
          }
        }
      });

      // 建筑：投影 → 屋顶 → 细节（投影用模糊模拟太阳软阴影）
      buildings.forEach(bd => {
        g.save();
        g.filter = 'blur(4px)';
        g.fillStyle = 'rgba(24,28,22,0.58)';
        g.fillRect(bd.x + 14, bd.y + 17, bd.w, bd.h);
        g.restore();
      });
      buildings.forEach(bd => {
        const tone = rint(0, 5);
        // 真实航拍里屋顶材质差异很大（水泥/沥青卷材/彩钢/瓦），
        // 用冷暖混合的色板打散"一片白"的观感。
        g.fillStyle = ['#d9d6cc', '#b7bcb8', '#c9a795', '#a9b0b6', '#cfc7b4', '#9daaa6'][tone];
        g.fillRect(bd.x, bd.y, bd.w, bd.h);
        g.strokeStyle = 'rgba(88,90,84,0.8)'; g.lineWidth = 1.5;
        g.strokeRect(bd.x + 0.5, bd.y + 0.5, bd.w - 1, bd.h - 1);
        // 女儿墙内圈（让屋顶有厚度感）
        g.strokeStyle = 'rgba(255,255,255,0.34)'; g.lineWidth = 1;
        g.strokeRect(bd.x + 4.5, bd.y + 4.5, bd.w - 9, bd.h - 9);
        // 屋顶受光面
        g.fillStyle = 'rgba(255,255,255,0.22)';
        g.fillRect(bd.x + 3, bd.y + 3, bd.w - 6, bd.h * 0.30);
        // 屋面接缝
        g.strokeStyle = 'rgba(104,104,98,0.28)'; g.lineWidth = 1;
        const seam = rint(2, 4);
        for (let i = 1; i <= seam; i++) {
          const sx = bd.x + (bd.w / (seam + 1)) * i;
          g.beginPath(); g.moveTo(sx, bd.y + 6); g.lineTo(sx, bd.y + bd.h - 6); g.stroke();
        }
        // 屋面设备（带自身小投影）
        const n = rint(1, 3);
        for (let i = 0; i < n; i++) {
          const vw = rf(9, Math.max(12, bd.w * 0.24)), vh = rf(9, Math.max(12, bd.h * 0.24));
          const vx = bd.x + rf(7, Math.max(8, bd.w - vw - 7));
          const vy = bd.y + rf(7, Math.max(8, bd.h - vh - 7));
          g.fillStyle = 'rgba(52,54,50,0.45)';
          g.fillRect(vx + 2.5, vy + 3, vw, vh);
          g.fillStyle = 'rgba(118,120,116,0.9)';
          g.fillRect(vx, vy, vw, vh);
          g.fillStyle = 'rgba(255,255,255,0.30)';
          g.fillRect(vx, vy, vw, vh * 0.34);
        }
      });

      /* --- 目标：车辆与行人 --- */
      const dets = [];
      const addDet = (cls, x, y, w, h, conf) => {
        dets.push({
          cls: cls, conf: +conf.toFixed(3),
          x1: +x.toFixed(1), y1: +y.toFixed(1), x2: +(x + w).toFixed(1), y2: +(y + h).toFixed(1),
          lon: null, lat: null,
        });
      };

      const VEHICLE = [
        { cls: 3, w: 52, h: 27, body: '#e8e8e6', roof: '#b8bcc2', conf: [0.72, 0.97] },  // 白车
        { cls: 3, w: 52, h: 27, body: '#3a3f46', roof: '#5d646e', conf: [0.70, 0.96] },  // 深色车
        { cls: 3, w: 52, h: 27, body: '#9aa0a8', roof: '#c2c8d0', conf: [0.68, 0.96] },  // 银车
        { cls: 3, w: 52, h: 27, body: '#8f3b32', roof: '#b25a4e', conf: [0.70, 0.95] },  // 红车
        { cls: 3, w: 52, h: 27, body: '#2f4d78', roof: '#4a6d9c', conf: [0.68, 0.95] },  // 蓝车
        { cls: 4, w: 62, h: 31, body: '#dcdcd6', roof: '#f0f0ea', conf: [0.66, 0.93] },  // 厢式货车
        { cls: 5, w: 84, h: 36, body: '#7d8288', roof: '#9aa0a6', conf: [0.64, 0.92] },  // 卡车
        { cls: 8, w: 96, h: 38, body: '#2f7a5e', roof: '#4a9c7c', conf: [0.70, 0.94] },  // 公交车
        { cls: 9, w: 32, h: 14, body: '#c8ccd0', roof: '#e2e6ea', conf: [0.42, 0.82] },  // 摩托车
        { cls: 2, w: 27, h: 13, body: '#6a7078', roof: '#8e949c', conf: [0.36, 0.74] },  // 自行车
        { cls: 6, w: 35, h: 23, body: '#7a7f5a', roof: '#9ba073', conf: [0.40, 0.78] },  // 三轮车
        { cls: 7, w: 39, h: 27, body: '#5a5566', roof: '#7d7689', conf: [0.38, 0.76] },  // 带棚三轮车
      ];

      function drawVehicle(x, y, v, horizontal) {
        const w = horizontal ? v.w : v.h;
        const h = horizontal ? v.h : v.w;
        // 投影（软阴影）
        g.save();
        g.filter = 'blur(2.4px)';
        g.fillStyle = 'rgba(18,22,16,0.46)';
        g.fillRect(x + 3.5, y + 4.5, w, h);
        g.restore();
        // 车身
        g.fillStyle = v.body;
        g.beginPath();
        const r = 3;
        g.moveTo(x + r, y);
        g.arcTo(x + w, y, x + w, y + h, r);
        g.arcTo(x + w, y + h, x, y + h, r);
        g.arcTo(x, y + h, x, y, r);
        g.arcTo(x, y, x + w, y, r);
        g.closePath(); g.fill();
        // 车身侧向渐变，避免纯色平板
        const lg = horizontal
          ? g.createLinearGradient(0, y, 0, y + h)
          : g.createLinearGradient(x, 0, x + w, 0);
        lg.addColorStop(0, 'rgba(255,255,255,0.22)');
        lg.addColorStop(0.5, 'rgba(255,255,255,0)');
        lg.addColorStop(1, 'rgba(0,0,0,0.26)');
        g.fillStyle = lg;
        g.beginPath();
        g.moveTo(x + r, y);
        g.arcTo(x + w, y, x + w, y + h, r);
        g.arcTo(x + w, y + h, x, y + h, r);
        g.arcTo(x, y + h, x, y, r);
        g.arcTo(x, y, x + w, y, r);
        g.closePath(); g.fill();
        // 车顶 / 挡风
        g.fillStyle = v.roof;
        if (horizontal) g.fillRect(x + w * 0.28, y + h * 0.16, w * 0.4, h * 0.68);
        else g.fillRect(x + w * 0.16, y + h * 0.28, w * 0.68, h * 0.4);
        // 前后挡风玻璃
        g.fillStyle = 'rgba(26,34,44,0.5)';
        if (horizontal) {
          g.fillRect(x + w * 0.09, y + h * 0.2, w * 0.14, h * 0.6);
          g.fillRect(x + w * 0.77, y + h * 0.2, w * 0.12, h * 0.6);
        } else {
          g.fillRect(x + w * 0.2, y + h * 0.09, w * 0.6, h * 0.14);
          g.fillRect(x + w * 0.2, y + h * 0.77, w * 0.6, h * 0.12);
        }
        // 高光
        g.fillStyle = 'rgba(255,255,255,0.30)';
        g.fillRect(x + 1, y + 1, w - 2, 1.6);
      }

      // 沿道路布置车辆
      roads.forEach(rd => {
        const horizontal = rd.dir === 'h';
        const lanes = horizontal
          ? [rd.y + rd.h * 0.26, rd.y + rd.h * 0.74]
          : [rd.x + rd.w * 0.26, rd.x + rd.w * 0.74];
        lanes.forEach(lane => {
          const count = horizontal ? rint(6, 10) : rint(4, 7);
          let t = rf(0, 60);
          for (let i = 0; i < count; i++) {
            const v = pick(VEHICLE);
            const span = horizontal ? W : H;
            const along = t;
            t += (horizontal ? v.w : v.h) + rf(70, 210);
            if (t > span - 40) break;
            const jitter = rf(-rd.h * 0.06, rd.h * 0.06);
            let x, y;
            if (horizontal) { x = along; y = lane - v.h / 2 + jitter; }
            else { x = lane - v.h / 2 + jitter; y = along; }
            drawVehicle(x, y, v, horizontal);
            const bw = horizontal ? v.w : v.h;
            const bh = horizontal ? v.h : v.w;
            addDet(v.cls, x, y, bw, bh, rf(v.conf[0], v.conf[1]));
          }
        });
      });

      // 行人 / 人群：人行道与斑马线附近
      const sidewalks = [];
      roads.forEach(rd => {
        if (rd.dir === 'h') {
          sidewalks.push({ x: 0, y: rd.y - 15, w: W, h: 15 });
          sidewalks.push({ x: 0, y: rd.y + rd.h, w: W, h: 15 });
        } else {
          sidewalks.push({ x: rd.x - 15, y: 0, w: 15, h: H });
          sidewalks.push({ x: rd.x + rd.w, y: 0, w: 15, h: H });
        }
      });

      function drawPerson(x, y, big) {
        const s = big ? 16 : 11;
        // 投影（软阴影）
        g.save();
        g.filter = 'blur(1.6px)';
        g.fillStyle = 'rgba(14,18,12,0.5)';
        g.beginPath(); g.ellipse(x + 2.4, y + 3.6, s * 0.48, s * 0.34, 0, 0, Math.PI * 2); g.fill();
        g.restore();
        // 躯干（俯视：肩部椭圆）
        g.fillStyle = big ? '#6d5a48' : '#48505e';
        g.beginPath(); g.ellipse(x, y + s * 0.06, s * 0.40, s * 0.30, 0, 0, Math.PI * 2); g.fill();
        // 头部
        g.fillStyle = big ? '#8a7256' : '#5b6473';
        g.beginPath(); g.arc(x, y - s * 0.22, s * 0.20, 0, Math.PI * 2); g.fill();
        // 顶部微高光，避免糊成一片
        g.fillStyle = 'rgba(255,255,255,0.20)';
        g.beginPath(); g.arc(x - s * 0.07, y - s * 0.27, s * 0.09, 0, Math.PI * 2); g.fill();
        return { w: s, h: s * 1.15 };
      }

      sidewalks.forEach(sw => {
        const n = rint(10, 22);
        for (let i = 0; i < n; i++) {
          const x = sw.x + rand() * sw.w;
          const y = sw.y + rand() * sw.h;
          const big = rand() < 0.18;
          const b = drawPerson(x, y, big);
          addDet(0, x - b.w / 2, y - b.h / 2, b.w, b.h, rf(big ? 0.48 : 0.30, big ? 0.88 : 0.72));
        }
      });

      // 人群：斑马线上聚簇
      [[600, 470], [1250, 830], [700, 420]].forEach(pt => {
        const n = rint(5, 11);
        for (let i = 0; i < n; i++) {
          const x = pt[0] + rf(-26, 26), y = pt[1] + rf(-26, 26);
          const b = drawPerson(x, y, false);
          addDet(1, x - b.w / 2, y - b.h / 2, b.w, b.h, rf(0.34, 0.8));
        }
      });

      // 少量低置信度目标，用于演示阈值过滤
      for (let i = 0; i < 14; i++) {
        const x = rf(40, W - 60), y = rf(40, H - 60);
        addDet(pick([0, 2, 3, 9]), x, y, rf(8, 22), rf(8, 18), rf(0.06, 0.24));
      }

      /* --- 降采样 + 成像质感 ------------------------------------------- */
      // 先高质量降采样，再在最终分辨率上叠加暗角、颗粒与 OSD，
      // 保证 OSD 文字锐利、颗粒细腻，不随超采样被抹平。
      const out = document.createElement('canvas');
      out.width = W; out.height = H;
      const og = out.getContext('2d');
      og.imageSmoothingEnabled = true;
      og.imageSmoothingQuality = 'high';
      // 注意：这里不要加 ctx.filter——带 filter 的 drawImage 会绕过高质量
      // 降采样路径，画面会明显发糊。软化交给超采样本身即可。
      og.drawImage(cv, 0, 0, W, H);

      // 暗角
      const vg = og.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.38, W / 2, H / 2, Math.max(W, H) * 0.78);
      vg.addColorStop(0, 'rgba(0,0,0,0)');
      vg.addColorStop(1, 'rgba(0,0,0,0.26)');
      og.fillStyle = vg; og.fillRect(0, 0, W, H);

      // 传感器颗粒：64×64 噪声瓦片平铺，比逐点噪声更快、颗粒更细
      const NT = 64;
      const nc = document.createElement('canvas');
      nc.width = NT; nc.height = NT;
      const ng = nc.getContext('2d');
      const nid = ng.createImageData(NT, NT);
      for (let i = 0; i < nid.data.length; i += 4) {
        const v = 128 + (rand() - 0.5) * 160;
        nid.data[i] = nid.data[i + 1] = nid.data[i + 2] = v;
        nid.data[i + 3] = 15;
      }
      ng.putImageData(nid, 0, 0);
      og.save();
      og.globalCompositeOperation = 'overlay';
      og.fillStyle = og.createPattern(nc, 'repeat');
      og.fillRect(0, 0, W, H);
      og.restore();

      // 顶部叠一行"采集信息"，模拟真实航拍图的 OSD
      og.fillStyle = 'rgba(10,14,18,0.62)';
      og.fillRect(0, 0, W, 34);
      og.fillStyle = 'rgba(224,236,248,0.86)';
      og.font = '600 15px ui-monospace, Consolas, monospace';
      og.fillText('VISDRONE-DET / UAV-04  ·  ALT 50.0m  ·  GSD 0.021 m/px  ·  10 CLASSES', 14, 22);
      og.fillStyle = 'rgba(224,236,248,0.6)';
      og.font = '600 13px ui-monospace, Consolas, monospace';
      const ts = new Date();
      const pp = n => String(n).padStart(2, '0');
      const stamp = ts.getFullYear() + '-' + pp(ts.getMonth() + 1) + '-' + pp(ts.getDate()) + ' ' +
        pp(ts.getHours()) + ':' + pp(ts.getMinutes()) + ':' + pp(ts.getSeconds());
      og.fillText(stamp, W - 232, 22);

      return {
        dataUrl: out.toDataURL('image/jpeg', 0.95),
        width: W, height: H,
        detections: dets,
        ms: Math.round(rf(28, 74)),
        engine: 'demo',
      };
    },

    /** 缩略图（用于胶片条） */
    thumb(dataUrl, maxW) {
      return new Promise(resolve => {
        const im = new Image();
        im.onload = () => {
          const k = (maxW || 136) / im.width;
          const c = document.createElement('canvas');
          c.width = Math.round(im.width * k);
          c.height = Math.round(im.height * k);
          c.getContext('2d').drawImage(im, 0, 0, c.width, c.height);
          resolve(c.toDataURL('image/jpeg', 0.7));
        };
        im.onerror = () => resolve(null);
        im.src = dataUrl;
      });
    },
  };
  BDS.Demo = Demo;

  /* ----------------------------------------------------------- 地理换算 */
  /**
   * 像素 → 经纬度。**与后端 geo_map 用同一个 GSD**（由接口下发，见 beidou.gsd_m_per_px）。
   *
   * 历史上这里是 `0.00001 * (w/640)` 度/像素（1920 宽 ≈ 2.94 m/px），
   * 而 ENU 那边写死 0.021 m/px —— 同一张图差约 140 倍，表格里两列数字互相矛盾。
   *
   * 没有有效定位、或 GSD 为 0（未标定且不允许估算）时返回 null，
   * 由调用方显示"—"。**不要退化成 0**：那会画出一个假坐标。
   */
  BDS.geoFromPixel = function (cx, cy, w, h) {
    const b = Store.state.beidou || {};
    const gsd = +b.gsd_m_per_px || 0;
    if (!gsd || b.lat == null || b.lon == null || !b.usable) return null;
    const enu = BDS.enuFromPixel(cx, cy, w, h);
    if (enu.e == null) return null;
    const M_PER_DEG_LAT = 111320.0;
    const cosLat = Math.cos(b.lat * Math.PI / 180);
    if (Math.abs(cosLat) < 1e-6) return null;
    return {
      lat: +(b.lat + enu.n / M_PER_DEG_LAT).toFixed(7),
      lon: +(b.lon + enu.e / (M_PER_DEG_LAT * cosLat)).toFixed(7),
    };
  };

  /** 像素 → 本地 ENU（东/北，单位米）。GSD 同样取自接口，不再写死常数。 */
  BDS.enuFromPixel = function (cx, cy, w, h) {
    const b = Store.state.beidou || {};
    const gsd = +b.gsd_m_per_px || 0;
    if (!gsd) return { e: null, n: null };
    return {
      e: +((cx - w / 2) * gsd).toFixed(2),
      n: +(-(cy - h / 2) * gsd).toFixed(2),
    };
  };

  /** 米 → 人类可读（本地散点轴的刻度用） */
  BDS.fmtMeters = function (v) {
    if (v == null) return '—';
    return (Math.abs(v) >= 1000 ? (v / 1000).toFixed(2) + ' km' : v.toFixed(1) + ' m');
  };

  /**
   * 经纬度 → 本地 ENU（东/北，单位米），基准是 (refLat, refLon)。
   *
   * 与 :func:`geoFromPixel` 严格互逆，用同一组常数：拖动时间轴时，
   * 轨迹折线（经纬度来的）与目标点（像素来的）必须落在同一套坐标里，
   * 否则两条线会差出一个比例尺 —— 这个坑在 geo_map / enuFromPixel 上踩过一次。
   */
  BDS.enuFromGeo = function (lat, lon, refLat, refLon) {
    if (lat == null || lon == null || refLat == null || refLon == null) return { e: null, n: null };
    const M_PER_DEG_LAT = 111320.0;
    const cosLat = Math.cos(refLat * Math.PI / 180);
    if (Math.abs(cosLat) < 1e-6) return { e: null, n: null };
    return {
      e: +((lon - refLon) * M_PER_DEG_LAT * cosLat).toFixed(2),
      n: +((lat - refLat) * M_PER_DEG_LAT).toFixed(2),
    };
  };

})(window);

/* 截图探针：通过 location.hash 驱动界面状态
 * 例：#view=analytics&theme=light&zoom=1.4&palette=导出&select=first&keys=1&about=1
 *
 * 异步场景（推理 / 拉取指标）用 wait=<ms>：探针会返回 Promise，
 * _shot.js 以 awaitPromise 等待，届时再回读 trainstat / detstat。
 * 例：#view=detect&run=1&wait=6000&detstat=1
 */
(function () {
  var h = (location.hash || '').replace(/^#/, '');
  var out = {};
  if (!h) return 'noop';

  /** 回读「训练成果」卡片的渲染结果（断言用，不依赖截图） */
  function readTrain() {
    var td = window.BDS && BDS.Train && BDS.Train.data;
    if (!td) return null;
    return {
      run: td.run,
      epochs: td.epochs,
      best: td.best,
      params: td.params,
      available: td.available,
      kpis: Array.prototype.map.call(
        document.querySelectorAll('#train-kpis .kpi .v'),
        function (e) { return e.textContent; }),
      curve: document.querySelector('#train-curve svg') ? 1 : 0,
      cmpRows: document.querySelectorAll('#train-cmp tbody tr').length,
      arts: document.querySelectorAll('#train-art figure').length,
    };
  }

  /** 回读当前帧的推理结果（判断是否真的走了后端） */
  function readDet() {
    var s = window.BDS && BDS.Store.state;
    if (!s) return null;
    var f = s.frames[s.activeId];
    if (!f) return null;
    return {
      done: !!f.done,
      n: (f.dets || []).length,
      engine: f.engine || null,
      device: f.device || null,
      ms: f.ms == null ? null : Math.round(f.ms),
      mode: s.model.mode,
      apiOnline: s.api.online,
      prefix: s.api.prefix,
    };
  }

  var waitMs = 0;

  h.split('&').forEach(function (p) {
    var i = p.indexOf('=');
    var k = i < 0 ? p : p.slice(0, i);
    var v = i < 0 ? '' : p.slice(i + 1);
    v = decodeURIComponent(v);
    try {
      if (k === 'view') {
        var b = document.querySelector('.rail-btn[data-view="' + v + '"]');
        if (b) { b.click(); out.view = v; }
      } else if (k === 'theme') {
        if (document.documentElement.getAttribute('data-theme') !== v) document.querySelector('#btn-theme').click();
        out.theme = document.documentElement.getAttribute('data-theme');
      } else if (k === 'zoom') {
        var s = parseFloat(v), st = document.querySelector('#stage');
        BDS.Overlay.view.scale = s;
        BDS.Overlay.view.x = st.clientWidth / 2 - 760 * s;
        BDS.Overlay.view.y = st.clientHeight / 2 - 430 * s;
        BDS.Overlay.render();
        document.querySelector('#zoom-val').textContent = Math.round(s * 100) + '%';
        out.zoom = s;
      } else if (k === 'palette') {
        document.querySelector('#cmd-open').click();
        var inp = document.querySelector('#pal-input');
        inp.value = v; inp.dispatchEvent(new Event('input'));
        out.palette = v;
      } else if (k === 'select') {
        var row = document.querySelector('#det-list .det-row');
        if (row) { row.click(); out.select = 'ok'; }
      } else if (k === 'keys') {
        document.querySelector('#rail-keys').click(); out.keys = 1;
      } else if (k === 'about') {
        document.querySelector('#rail-about').click(); out.about = 1;
      } else if (k === 'toast') {
        BDS.toast('ok', '推理完成', '195 个目标 · 48 ms');
        BDS.toast('warn', '后端未连接', '界面以离线演示形态运行');
        out.toast = 1;
      } else if (k === 'mask') {
        ['#tg-mask', '#tg-index', '#tg-compare'].forEach(function (sel) {
          var e = document.querySelector(sel); if (e) e.click();
        });
        out.mask = 1;
      } else if (k === 'params') {
        BDS.Store.state.params.conf = 0.55;
        document.querySelector('#p-conf').value = 0.55;
        document.querySelector('#p-conf').dispatchEvent(new Event('input'));
        document.querySelector('#p-slice').click();
        out.params = 1;
      } else if (k === 'hidecls') {
        var rows = document.querySelectorAll('#dist .dist-row');
        if (rows[0]) rows[0].click();
        if (rows[1]) rows[1].click();
        out.hidecls = 1;
      } else if (k === 'drawer') {
        // r = 切右栏（窄屏抽屉 / 宽屏收起整列），l = 切左栏抽屉
        var db = v === 'l' ? document.querySelector('#btn-panel-l') : document.querySelector('#btn-panel-r');
        if (db) db.click();
        out.drawer = v;
      } else if (k === 'mockjobs') {
        // 注入四种状态各一份假任务，用于在没有后端时验收"有数据态"的任务视图
        var now = Date.now();
        var mk = function (i, st, name, done, total, stats, err) {
          return {
            id: 'a' + i + 'f3c9e2-7b41-4d8a-9e15-' + (100000000000 + i * 137),
            status: st, name: name, size: 40000000 + i * 7300000,
            createdAt: now - i * 37 * 60000,
            progress: { done: done, total: total },
            params: { conf: 0.25, iou: 0.45, filename: name },
            result: stats ? { stats: stats } : {},
            error: err || null,
          };
        };
        BDS.Store.state.jobs = [
          mk(1, 'processing', '无人机巡检_滨江路_04.mp4', 812, 1440, null, null),
          mk(2, 'completed', '交通路口_早高峰_02.mp4', 1440, 1440,
            { total: 18426, frames: 1440, avg_per_frame: 12.8, max_per_frame: 47 }, null),
          mk(3, 'queued', '园区周界_夜巡_07.mp4', 0, 2160, null, null),
          mk(4, 'failed', '工地_塔吊_01.mp4', 214, 980, null,
            '解码失败：第 214 帧 H.265 关键帧损坏（ffmpeg 返回非零退出码）'),
          mk(5, 'completed', '校园_午后_03.mp4', 900, 900,
            { total: 9310, frames: 900, avg_per_frame: 10.3, max_per_frame: 38 }, null),
        ];
        out.mockjobs = BDS.Store.state.jobs.length;
      } else if (k === 'jobsel') {
        var jrows = document.querySelectorAll('#job-list .job-row');
        var ji = parseInt(v, 10) || 0;
        if (jrows[ji]) { jrows[ji].click(); out.jobsel = ji; }
      } else if (k === 'run') {
        document.querySelector('#btn-run').click();
        out.run = 1;
      } else if (k === 'train') {
        // 滚到「训练成果」卡片，便于截屏取景
        var tk = document.querySelector('#train-kpis');
        if (tk) tk.scrollIntoView({ block: 'start' });
        out.train = !!tk;
      } else if (k === 'trainstat') {
        out.trainstat = readTrain();
      } else if (k === 'detstat') {
        out.detstat = readDet();
      } else if (k === 'lb') {
        // 点开某张训练配图
        var fg = document.querySelector('#train-art figure[data-art="' + v + '"]');
        if (fg) fg.click();
        out.lb = !!fg;
      } else if (k === 'state') {
        var st2 = window.BDS && BDS.Store.state;
        out.state = st2 ? {
          view: st2.view, activeId: st2.activeId, queue: st2.queue.length,
          frames: Object.keys(st2.frames).length, running: st2.running,
          apiOnline: st2.api.online, prefix: st2.api.prefix,
        } : null;
      } else if (k === 'spytoast') {
        // 记录所有 toast 调用，便于定位"静默失败"
        if (!window.__toasts) {
          window.__toasts = [];
          var origToast = BDS.toast;
          BDS.toast = function () {
            window.__toasts.push(Array.prototype.slice.call(arguments));
            return origToast.apply(this, arguments);
          };
        }
        out.spytoast = 1;
      } else if (k === 'callrun') {
        // 直接调用推理入口（绕开按钮）。演示场景是无头软件渲染的瓶颈，
        // 所以先轮询等帧就绪，再发起推理，否则会读到"没有可推理的输入"。
        out.callrun = typeof (window.BDS && BDS.runInference);
        var tries = 0;
        var iv = setInterval(function () {
          tries++;
          var stc = BDS.Store.state;
          if (stc.activeId && stc.frames[stc.activeId]) {
            clearInterval(iv);
            out.demoTicks = tries;
            BDS.runInference().then(
              function () { out.runDone = readDet(); },
              function (e) { out.runErr = String(e); });
          } else if (tries > 160) {
            clearInterval(iv);
            out.demoTimeout = true;
          }
        }, 500);
      } else if (k === 'wait') {
        waitMs = parseInt(v, 10) || 0;
        out.wait = waitMs;
      }
    } catch (e) { out['err_' + k] = String(e); }
  });

  if (waitMs > 0) {
    // 有异步动作（推理 / 拉指标）时轮询等待：动作落定即返回，或到 waitMs 上限
    return new Promise(function (res) {
      var t0 = Date.now();
      var iv2 = setInterval(function () {
        var settled = !!(out.runDone || out.runErr || out.demoTimeout);
        if (settled || Date.now() - t0 > waitMs) {
          clearInterval(iv2);
          out.waitedMs = Date.now() - t0;
          out.trainstat = readTrain();
          out.detstat = readDet();
          var st3 = window.BDS && BDS.Store.state;
          if (st3) {
            out.endState = {
              activeId: st3.activeId, queue: st3.queue.length,
              frames: Object.keys(st3.frames).length, running: st3.running,
            };
          }
          if (window.__toasts) out.toasts = window.__toasts;
          res(JSON.stringify(out));
        }
      }, 400);
    });
  }
  return JSON.stringify(out);
})();

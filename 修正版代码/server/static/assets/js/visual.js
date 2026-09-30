/* ============================================================================
 * visual.js — 可视化层
 *   · Overlay  高 DPI 画布标注渲染器（缩放 / 平移 / 命中测试）
 *   · Charts   零依赖 SVG 图表（环形 / 条形 / 趋势 / 直方图）
 *   · Beidou   卫星星空图（方位角-高度角极坐标）与 ENU 目标散布图
 * ========================================================================== */
(function (global) {
  'use strict';

  const BDS = global.BDS;
  const U = BDS.util;

  /* =========================================================== Overlay */
  const Overlay = {
    canvas: null,
    ctx: null,
    stage: null,
    img: null,
    dets: [],
    view: { scale: 1, x: 0, y: 0 },
    opts: {},

    init(canvas, stage) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.stage = stage;
    },

    setImage(img) {
      this.img = img;
      if (img) this.fit();
    },

    /** 计算适应窗口的缩放与居中偏移 */
    fit() {
      if (!this.img || !this.stage) return;
      const sw = this.stage.clientWidth, sh = this.stage.clientHeight;
      // 舞台不可见（视图切换 / 布局未就绪 / 面板折叠）时尺寸为 0，
      // 此时任何计算都会得到负缩放，直接跳过，等下一次有效尺寸再算。
      if (sw < 8 || sh < 8) return;
      const pad = 24;
      const k = Math.min((sw - pad * 2) / this.img.width, (sh - pad * 2) / this.img.height, 1);
      if (!(k > 0) || !isFinite(k)) return;
      this.view.scale = k;
      this.view.x = (sw - this.img.width * k) / 2;
      this.view.y = (sh - this.img.height * k) / 2;
      this.render();
    },

    /** 以屏幕坐标 (sx,sy) 为锚点缩放 */
    zoomAt(sx, sy, factor) {
      const v = this.view;
      const next = U.clamp(v.scale * factor, 0.04, 24);
      const real = next / v.scale;
      v.x = sx - (sx - v.x) * real;
      v.y = sy - (sy - v.y) * real;
      v.scale = next;
      this.render();
    },

    panBy(dx, dy) { this.view.x += dx; this.view.y += dy; this.render(); },

    toImage(sx, sy) {
      const v = this.view;
      return { x: (sx - v.x) / v.scale, y: (sy - v.y) / v.scale };
    },

    toScreen(ix, iy) {
      const v = this.view;
      return { x: ix * v.scale + v.x, y: iy * v.scale + v.y };
    },

    /** 命中测试：返回最上层（面积最小者优先）的目标下标 */
    hit(sx, sy) {
      const p = this.toImage(sx, sy);
      const list = this.opts.visible || [];
      let best = -1, bestArea = Infinity;
      for (let i = 0; i < this.dets.length; i++) {
        const d = this.dets[i];
        if (list.indexOf(d.cls) < 0) continue;
        if (p.x >= d.x1 && p.x <= d.x2 && p.y >= d.y1 && p.y <= d.y2) {
          const a = (d.x2 - d.x1) * (d.y2 - d.y1);
          if (a < bestArea) { bestArea = a; best = i; }
        }
      }
      return best;
    },

    /** 将某个目标居中放大到合适倍率 */
    focusOn(idx) {
      const d = this.dets[idx];
      if (!d || !this.stage) return;
      const sw = this.stage.clientWidth, sh = this.stage.clientHeight;
      const bw = Math.max(6, d.x2 - d.x1), bh = Math.max(6, d.y2 - d.y1);
      const k = U.clamp(Math.min(sw / (bw * 4.2), sh / (bh * 4.2)), 0.08, 12);
      this.view.scale = k;
      this.view.x = sw / 2 - ((d.x1 + d.x2) / 2) * k;
      this.view.y = sh / 2 - ((d.y1 + d.y2) / 2) * k;
      this.render();
    },

    resize() {
      if (!this.stage) return;
      const w = this.stage.clientWidth, h = this.stage.clientHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = Math.max(1, Math.round(w * dpr));
      this.canvas.height = Math.max(1, Math.round(h * dpr));
      this.canvas.style.width = w + 'px';
      this.canvas.style.height = h + 'px';
      this.dpr = dpr;
      this.render();
    },

    render() {
      const ctx = this.ctx;
      if (!ctx) return;
      const o = this.opts;
      const st = BDS.Store.state;

      const w = this.canvas.width / (this.dpr || 1);
      const h = this.canvas.height / (this.dpr || 1);
      ctx.setTransform(this.dpr || 1, 0, 0, this.dpr || 1, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (!this.img || !(this.view.scale > 0)) return;

      const v = this.view;
      const visible = o.visible || BDS.CLASSES.map(c => c.id);
      const colorOf = BDS.classColor;
      const sel = o.selected == null ? -1 : o.selected;
      const hov = o.hover == null ? -1 : o.hover;

      // 目标排序：大框先画，小框后画（保证小目标可点）
      const order = this.dets.map((d, i) => i).filter(i => visible.indexOf(this.dets[i].cls) >= 0);
      order.sort((a, b) => {
        const A = this.dets[a], B = this.dets[b];
        return (B.x2 - B.x1) * (B.y2 - B.y1) - (A.x2 - A.x1) * (A.y2 - A.y1);
      });

      ctx.save();
      ctx.translate(v.x, v.y);
      ctx.scale(v.scale, v.scale);

      // 原图
      ctx.imageSmoothingEnabled = v.scale < 1;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(this.img, 0, 0);

      // 图像描边
      ctx.save();
      ctx.strokeStyle = 'rgba(140,170,210,0.30)';
      ctx.lineWidth = 1 / v.scale;
      ctx.strokeRect(0, 0, this.img.width, this.img.height);
      ctx.restore();

      // 参考栅格
      if (st.ui.grid) {
        ctx.save();
        ctx.lineWidth = 1 / v.scale;
        ctx.strokeStyle = 'rgba(120,160,220,0.16)';
        const step = 100;
        for (let x = step; x < this.img.width; x += step) {
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, this.img.height); ctx.stroke();
        }
        for (let y = step; y < this.img.height; y += step) {
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(this.img.width, y); ctx.stroke();
        }
        ctx.restore();
      }

      // 对比模式：遮住右半边，露出原图
      if (st.ui.compare) {
        ctx.save();
        ctx.fillStyle = 'rgba(0,0,0,0.001)';
        ctx.fillRect(0, 0, 0, 0);
        ctx.restore();
      }

      const sc = v.scale;
      if (!(sc > 0) || !isFinite(sc)) return;
      const lw = 1.6 / sc;
      const tick = Math.min(16, Math.max(7, 14)) / sc;
      const fam = getComputedStyle(document.body).fontFamily;

      // ---- 标签预排布 ----------------------------------------------------
      // 密集场景下逐个「贴在框上方」会大量互相压字。这里先按优先级（选中 >
      // 悬停 > 小目标优先）为每个目标挑选一个不与已占用区域冲突的位置，
      // 依次尝试：框上方 / 框下方 / 框内顶部 / 框右侧 / 框左侧，全部冲突则
      // 不画该标签（选中与悬停目标强制保留）。位置与文本缓存在 labelPlan。
      const labelPlan = new Map();
      if (st.ui.labels) {
        const fsL = 12 / sc, padX = 5 / sc, padY = 3.5 / sc, gap = 2 / sc;
        const thL = fsL + padY * 2;
        const placed = [];
        ctx.font = '600 ' + fsL + 'px ' + fam;

        const pri = order.slice().sort((a, b) => {
          const rank = k => (k === sel ? 0 : k === hov ? 1 : 2);
          const ra = rank(a), rb = rank(b);
          if (ra !== rb) return ra - rb;
          const A = this.dets[a], B = this.dets[b];
          return (A.x2 - A.x1) * (A.y2 - A.y1) - (B.x2 - B.x1) * (B.y2 - B.y1);
        });

        for (let n = 0; n < pri.length; n++) {
          const i = pri[n];
          const d = this.dets[i];
          const isSel = i === sel, isHov = i === hov;
          const bw = d.x2 - d.x1, bh = d.y2 - d.y1;
          const scrW = bw * sc, scrH = bh * sc;
          const tiny = scrW < 11 || scrH < 11;
          if ((tiny || scrW <= 20) && !isSel && !isHov) continue;
          if (!(scrH > 9 || isSel || isHov)) continue;

          const cls = BDS.CLASSES[d.cls] || { zh: '?' };
          let txt = cls.zh;
          if (st.ui.conf) txt += '  ' + Math.round(d.conf * 100) + '%';
          if (st.ui.index) txt = '#' + (i + 1) + '  ' + txt;
          const tw = ctx.measureText(txt).width + padX * 2;

          const cands = [
            [d.x1, d.y1 - thL - gap, 'b'],
            [d.x1, d.y2 + gap, 't'],
            [d.x1, d.y1 + gap, 'b'],
            [d.x2 + gap, d.y1, 'b'],
            [d.x1 - tw - gap, d.y1, 'b'],
            [d.x2 - tw, d.y1 - thL - gap, 'b'],
          ];
          let pick = null;
          for (let c = 0; c < cands.length; c++) {
            const lx = clampImg(cands[c][0], this.img.width - tw);
            const ly = clampImg(cands[c][1], this.img.height - thL);
            const r = { x: lx, y: ly, w: tw, h: thL, ptr: cands[c][2] };
            if (!rectHits(r, placed, 1.5 / sc)) { pick = r; break; }
          }
          if (!pick && (isSel || isHov)) {
            pick = {
              x: clampImg(d.x1, this.img.width - tw),
              y: clampImg(d.y1 - thL - gap, this.img.height - thL),
              w: tw, h: thL, ptr: 'b',
            };
          }
          if (pick) { pick.txt = txt; labelPlan.set(i, pick); placed.push(pick); }
        }
      }

      for (let n = 0; n < order.length; n++) {
        const i = order[n];
        const d = this.dets[i];
        const col = colorOf(d.cls);
        const x = d.x1, y = d.y1, bw = d.x2 - d.x1, bh = d.y2 - d.y1;
        const isSel = i === sel, isHov = i === hov;
        const dim = sel >= 0 && !isSel;
        const scrW = bw * sc, scrH = bh * sc;
        // 屏幕尺度下的"微小目标"：改用点标记，避免细框糊成一团
        const tiny = scrW < 11 || scrH < 11;

        ctx.save();
        ctx.globalAlpha = dim ? 0.32 : 1;

        // 遮罩
        if (st.ui.mask) {
          ctx.fillStyle = hexA(col, isSel ? 0.30 : 0.17);
          ctx.fillRect(x, y, bw, bh);
        } else if (isSel) {
          ctx.fillStyle = hexA(col, 0.14);
          ctx.fillRect(x, y, bw, bh);
        }

        // 外发光（选中 / 悬停）
        if (isSel || isHov) {
          ctx.save();
          ctx.shadowColor = col;
          ctx.shadowBlur = (isSel ? 14 : 9) / sc;
          ctx.strokeStyle = col;
          ctx.lineWidth = (isSel ? 2.4 : 2) / sc;
          ctx.strokeRect(x, y, bw, bh);
          ctx.restore();
        }

        if (st.ui.boxes && (!tiny || isSel || isHov)) {
          // 主框
          ctx.strokeStyle = isSel ? '#ffffff' : col;
          ctx.lineWidth = (isSel ? 2.2 : lw);
          if (dim) ctx.strokeStyle = hexA(col, 0.75);
          ctx.strokeRect(x, y, bw, bh);

          // 四角加粗（专业标注工具范式）
          ctx.strokeStyle = col;
          ctx.lineWidth = (isSel ? 3.4 : 2.8) / sc;
          ctx.beginPath();
          // 左上
          ctx.moveTo(x, y + tick); ctx.lineTo(x, y); ctx.lineTo(x + tick, y);
          // 右上
          ctx.moveTo(x + bw - tick, y); ctx.lineTo(x + bw, y); ctx.lineTo(x + bw, y + tick);
          // 右下
          ctx.moveTo(x + bw, y + bh - tick); ctx.lineTo(x + bw, y + bh); ctx.lineTo(x + bw - tick, y + bh);
          // 左下
          ctx.moveTo(x + tick, y + bh); ctx.lineTo(x, y + bh); ctx.lineTo(x, y + bh - tick);
          ctx.stroke();

          // 置信度条（贴底边内侧）
          if (st.ui.conf) {
            const barH = Math.min(3.2 / sc, bh * 0.22);
            ctx.fillStyle = hexA(col, 0.28);
            ctx.fillRect(x, y + bh - barH, bw, barH);
            ctx.fillStyle = col;
            ctx.fillRect(x, y + bh - barH, bw * U.clamp(d.conf, 0, 1), barH);
          }
        }

        // 微小目标：点标记（空心环 + 实心点）
        if (tiny) {
          const mx = x + bw / 2, my = y + bh / 2;
          ctx.beginPath();
          ctx.arc(mx, my, (isSel ? 7 : 5.5) / sc, 0, Math.PI * 2);
          ctx.strokeStyle = hexA(col, isSel ? 0.95 : 0.5);
          ctx.lineWidth = 1.1 / sc;
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(mx, my, (isSel ? 3.4 : 2.6) / sc, 0, Math.PI * 2);
          ctx.fillStyle = col;
          ctx.fill();
        }

        // 标签（位置已在预排布阶段确定）
        const lp = labelPlan.get(i);
        if (lp) {
          const fs = 12 / sc, padX = 5 / sc;
          ctx.font = '600 ' + fs + 'px ' + fam;
          ctx.fillStyle = isSel ? '#ffffff' : col;
          roundRect(ctx, lp.x, lp.y, lp.w, lp.h, 3 / sc);
          ctx.fill();
          // 指向三角：标签在框上方则朝下，在框下方则朝上
          const ax = lp.x + 5 / sc;
          ctx.beginPath();
          if (lp.ptr === 't') {
            ctx.moveTo(ax, lp.y);
            ctx.lineTo(ax + 6 / sc, lp.y);
            ctx.lineTo(ax, lp.y - 4 / sc);
          } else {
            ctx.moveTo(ax, lp.y + lp.h);
            ctx.lineTo(ax + 6 / sc, lp.y + lp.h);
            ctx.lineTo(ax, lp.y + lp.h + 4 / sc);
          }
          ctx.closePath();
          ctx.fill();

          ctx.fillStyle = isSel ? '#0b0e12' : '#08101a';
          ctx.textBaseline = 'middle';
          ctx.fillText(lp.txt, lp.x + padX, lp.y + lp.h / 2 + 0.5 / sc);
        }

        ctx.restore();
      }

      // 北斗参考点（图像中心）
      const cx = this.img.width / 2, cy = this.img.height / 2;
      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.strokeStyle = BDS.classColor(0) === '' ? '#8b7cff' : '#8b7cff';
      ctx.lineWidth = 1.4 / sc;
      ctx.setLineDash([5 / sc, 5 / sc]);
      ctx.beginPath(); ctx.arc(cx, cy, 16 / sc, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(cx - 24 / sc, cy); ctx.lineTo(cx - 6 / sc, cy);
      ctx.moveTo(cx + 6 / sc, cy); ctx.lineTo(cx + 24 / sc, cy);
      ctx.moveTo(cx, cy - 24 / sc); ctx.lineTo(cx, cy - 6 / sc);
      ctx.moveTo(cx, cy + 6 / sc); ctx.lineTo(cx, cy + 24 / sc);
      ctx.stroke();
      ctx.restore();

      ctx.restore();

      // 对比模式：右半侧叠加原图
      if (st.ui.compare) {
        const imgW = this.img.width * v.scale;
        const splitX = v.x + imgW * 0.5;
        ctx.save();
        ctx.beginPath();
        ctx.rect(splitX, 0, w - splitX, h);
        ctx.clip();
        ctx.translate(v.x, v.y);
        ctx.scale(v.scale, v.scale);
        ctx.drawImage(this.img, 0, 0);
        ctx.restore();

        ctx.save();
        ctx.strokeStyle = '#4d9eff';
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(splitX, 0); ctx.lineTo(splitX, h); ctx.stroke();
        ctx.fillStyle = '#4d9eff';
        roundRect(ctx, splitX - 30, 12, 60, 20, 4); ctx.fill();
        ctx.fillStyle = '#04121f';
        ctx.font = '600 11px ' + getComputedStyle(document.body).fontFamily;
        ctx.textBaseline = 'middle';
        ctx.fillText('原图', splitX - 14, 22.5);
        ctx.restore();
      }
    },
  };

  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  /** 把标签左上角坐标夹进图像范围，保证标签完整可见 */
  function clampImg(v, max) {
    return Math.max(0, Math.min(max, v));
  }

  /** 矩形碰撞检测（grow 为外扩余量，制造呼吸间距） */
  function rectHits(r, list, grow) {
    const g = grow || 0;
    for (let i = 0; i < list.length; i++) {
      const o = list[i];
      if (r.x < o.x + o.w + g && r.x + r.w + g > o.x &&
          r.y < o.y + o.h + g && r.y + r.h + g > o.y) return true;
    }
    return false;
  }

  /** hex → rgba */
  function hexA(hex, a) {
    const h = String(hex).replace('#', '');
    const n = h.length === 3 ? h.split('').map(c => c + c).join('') : h;
    const r = parseInt(n.slice(0, 2), 16), g = parseInt(n.slice(2, 4), 16), b = parseInt(n.slice(4, 6), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
  }

  BDS.Overlay = Overlay;
  BDS.hexA = hexA;

  /* =========================================================== Charts */
  const NS = 'http://www.w3.org/2000/svg';
  /**
   * 创建 SVG 元素。
   * 注意：CSS 变量不能可靠地写在 presentation attribute 里（fill="var(--x)" 会被判为非法值），
   * 因此凡是取值含 var(...) 的一律改写为 style 属性。
   */
  function svgEl(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    if (attrs) {
      for (const k in attrs) {
        const v = attrs[k];
        if (typeof v === 'string' && v.indexOf('var(') >= 0) e.style.setProperty(k, v);
        else e.setAttribute(k, v);
      }
    }
    return e;
  }
  /** 自适应宽度的图表 svg（等比缩放，不拉伸） */
  function responsiveSvg(viewW, viewH, extra) {
    const a = { viewBox: '0 0 ' + viewW + ' ' + viewH, class: 'chart-svg', style: 'width:100%;height:auto;display:block' };
    if (extra) for (const k in extra) a[k] = extra[k];
    return svgEl('svg', a);
  }
  function polar(cx, cy, r, deg) {
    const a = (deg - 90) * Math.PI / 180;
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  }
  function arcPath(cx, cy, rOut, rIn, a0, a1) {
    const p0 = polar(cx, cy, rOut, a0), p1 = polar(cx, cy, rOut, a1);
    const p2 = polar(cx, cy, rIn, a1), p3 = polar(cx, cy, rIn, a0);
    const large = (a1 - a0) > 180 ? 1 : 0;
    return ['M', p0.x, p0.y, 'A', rOut, rOut, 0, large, 1, p1.x, p1.y,
      'L', p2.x, p2.y, 'A', rIn, rIn, 0, large, 0, p3.x, p3.y, 'Z'].join(' ');
  }

  const Charts = {
    /** 环形图 + 图例 */
    donut(host, items, opts) {
      opts = opts || {};
      const size = opts.size || 168, thick = opts.thickness || 16;
      const cx = size / 2, cy = size / 2, rOut = size / 2 - 2, rIn = rOut - thick;
      const total = items.reduce((s, it) => s + it.value, 0);
      const svg = svgEl('svg', { viewBox: '0 0 ' + size + ' ' + size, width: size, height: size, class: 'chart-svg' });

      // 底环
      svg.appendChild(svgEl('circle', {
        cx: cx, cy: cy, r: (rOut + rIn) / 2, fill: 'none',
        stroke: 'var(--panel-3)', 'stroke-width': thick,
      }));

      if (total > 0) {
        let a = -90;
        const gap = items.filter(i => i.value > 0).length > 1 ? 1.6 : 0;
        items.forEach(it => {
          if (it.value <= 0) return;
          const span = (it.value / total) * 360;
          const a0 = a + gap / 2, a1 = a + span - gap / 2;
          if (a1 > a0) {
            const p = svgEl('path', {
              d: arcPath(cx, cy, rOut, rIn, a0, a1),
              fill: it.color,
              opacity: 0.92,
            });
            const t = svgEl('title');
            t.textContent = it.label + '：' + it.value;
            p.appendChild(t);
            svg.appendChild(p);
          }
          a += span;
        });
      }

      const tv = svgEl('text', {
        x: cx, y: cy - 4, 'text-anchor': 'middle',
        fill: 'var(--fg)', 'font-size': 22, 'font-weight': 600,
        'font-family': 'var(--font-mono)',
      });
      tv.textContent = opts.centerValue != null ? opts.centerValue : total;
      svg.appendChild(tv);
      const tl = svgEl('text', {
        x: cx, y: cy + 14, 'text-anchor': 'middle',
        fill: 'var(--fg-3)', 'font-size': 10.5,
      });
      tl.textContent = opts.centerLabel || '目标总数';
      svg.appendChild(tl);

      const wrap = document.createElement('div');
      wrap.className = 'chart-flex';
      wrap.appendChild(svg);

      const lg = document.createElement('div');
      lg.className = 'chart-legend';
      const sorted = items.slice().sort((a, b) => b.value - a.value);
      sorted.forEach(it => {
        const row = document.createElement('div');
        row.className = 'chart-legend-row' + (it.value === 0 ? ' is-zero' : '');
        row.innerHTML =
          '<i class="sw" style="background:' + it.color + '"></i>' +
          '<span class="nm">' + U.esc(it.label) + '</span>' +
          '<span class="vl">' + it.value + '</span>' +
          '<span class="pc">' + (total ? Math.round(it.value / total * 100) : 0) + '%</span>';
        lg.appendChild(row);
      });
      wrap.appendChild(lg);

      host.innerHTML = '';
      host.appendChild(wrap);
    },

    /** 横向条形图 */
    hbar(host, items, opts) {
      opts = opts || {};
      const rowH = opts.rowH || 24;
      const labelW = opts.labelW || 74;
      const valueW = 42;
      const width = opts.width || 560;
      const max = Math.max(1, ...items.map(i => i.value));
      const h = items.length * rowH + 4;
      const svg = svgEl('svg', {
        viewBox: '0 0 ' + width + ' ' + h, width: '100%', height: h,
        preserveAspectRatio: 'none', class: 'chart-svg',
      });

      items.forEach((it, i) => {
        const y = i * rowH + 2;
        const barX = labelW;
        const barW = width - labelW - valueW;
        const w = Math.max(it.value > 0 ? 3 : 0, (it.value / max) * barW);

        const lb = svgEl('text', {
          x: labelW - 10, y: y + rowH / 2 + 1, 'text-anchor': 'end',
          fill: 'var(--fg-2)', 'font-size': 11.5, 'dominant-baseline': 'middle',
        });
        lb.textContent = it.label;
        svg.appendChild(lb);

        svg.appendChild(svgEl('rect', {
          x: barX, y: y + rowH / 2 - 5, width: barW, height: 10,
          rx: 3, fill: 'var(--panel-3)',
        }));

        const r = svgEl('rect', {
          x: barX, y: y + rowH / 2 - 5, width: w, height: 10,
          rx: 3, fill: it.color, opacity: 0.95,
        });
        const t = svgEl('title');
        t.textContent = it.label + '：' + it.value;
        r.appendChild(t);
        svg.appendChild(r);

        const vl = svgEl('text', {
          x: width - 4, y: y + rowH / 2 + 1, 'text-anchor': 'end',
          fill: 'var(--fg-2)', 'font-size': 11.5, 'font-family': 'var(--font-mono)',
          'dominant-baseline': 'middle',
        });
        vl.textContent = it.value;
        svg.appendChild(vl);
      });

      host.innerHTML = '';
      host.appendChild(svg);
    },

    /** 面积趋势线 */
    sparkline(host, values, opts) {
      opts = opts || {};
      const width = opts.width || 640, height = opts.height || 150;
      const padL = 42, padR = 12, padT = 14, padB = 22;
      const iw = width - padL - padR, ih = height - padT - padB;
      const svg = responsiveSvg(width, height);

      const n = values.length;
      // 值域策略：波动大时从 0 起，波动小时收紧到 [min,max] 附近，
      // 否则一条 44ms 的平线会贴着顶边、看起来像"没有数据"。
      const vmax = n ? Math.max.apply(null, values) : 0;
      const vmin = n ? Math.min.apply(null, values) : 0;
      let min = 0, max = Math.max(1, vmax) * 1.15;
      if (n > 1 && (vmax - vmin) < vmax * 0.4) {
        const mid = (vmax + vmin) / 2;
        const half = Math.max((vmax - vmin) * 0.9, vmax * 0.08, 1);
        min = Math.max(0, mid - half);
        max = mid + half;
      }
      const px = i => padL + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw);
      const py = v => padT + ih - ((v - min) / (max - min)) * ih;

      // 网格 + Y 轴刻度
      for (let k = 0; k <= 4; k++) {
        const v = min + (max - min) * (k / 4);
        const y = py(v);
        svg.appendChild(svgEl('line', {
          x1: padL, y1: y, x2: width - padR, y2: y, stroke: 'var(--line)', 'stroke-width': 1,
        }));
        const lb = svgEl('text', {
          x: padL - 8, y: y + 3.5, 'text-anchor': 'end',
          fill: 'var(--fg-4)', 'font-size': 9.5, 'font-family': 'var(--font-mono)',
        });
        lb.textContent = Math.round(v) + (opts.unit || '');
        svg.appendChild(lb);
      }

      if (n === 0) {
        const t = svgEl('text', {
          x: width / 2, y: height / 2, 'text-anchor': 'middle',
          fill: 'var(--fg-4)', 'font-size': 11.5,
        });
        t.textContent = '暂无数据';
        svg.appendChild(t);
      } else {
        const line = [], area = [];
        values.forEach((v, i) => { line.push([px(i), py(v)]); });
        area.push(['M', padL, py(min)]);
        line.forEach(p => area.push(['L', p[0], p[1]]));
        area.push(['L', px(n - 1), py(min)], ['Z']);

        const defs = svgEl('defs');
        const lg = svgEl('linearGradient', { id: 'grad-spark', x1: 0, y1: 0, x2: 0, y2: 1 });
        lg.appendChild(svgEl('stop', { offset: '0%', 'stop-color': 'var(--accent)', 'stop-opacity': 0.34 }));
        lg.appendChild(svgEl('stop', { offset: '100%', 'stop-color': 'var(--accent)', 'stop-opacity': 0.02 }));
        defs.appendChild(lg);
        svg.appendChild(defs);

        svg.appendChild(svgEl('path', {
          d: area.map(a => a.join(' ')).join(' '),
          fill: 'url(#grad-spark)', stroke: 'none',
        }));
        svg.appendChild(svgEl('path', {
          d: line.map((p, i) => (i ? 'L' : 'M') + p[0] + ' ' + p[1]).join(' '),
          fill: 'none', stroke: 'var(--accent)', 'stroke-width': 1.8,
          'stroke-linejoin': 'round', 'stroke-linecap': 'round',
        }));

        // 峰值标注
        let mi = 0;
        values.forEach((v, i) => { if (v > values[mi]) mi = i; });
        svg.appendChild(svgEl('circle', {
          cx: px(mi), cy: py(values[mi]), r: 3.2,
          fill: 'var(--panel)', stroke: 'var(--accent)', 'stroke-width': 1.8,
        }));
        const pl = svgEl('text', {
          x: px(mi), y: py(values[mi]) - 8, 'text-anchor': 'middle',
          fill: 'var(--fg-2)', 'font-size': 10, 'font-family': 'var(--font-mono)',
        });
        pl.textContent = Math.round(values[mi]) + (opts.unit || '');
        svg.appendChild(pl);

        // 最近一次
        svg.appendChild(svgEl('circle', {
          cx: px(n - 1), cy: py(values[n - 1]), r: 2.6,
          fill: 'var(--accent)',
        }));
      }

      host.innerHTML = '';
      host.appendChild(svg);
    },

    /**
     * 多序列折线图（训练收敛曲线 / 指标对比）。
     *
     * series: [{ name, values, color, dash }]
     * opts:   { width, height, fmt, yMin, yMax, yFloor0, xTicks, xStart,
     *           markBest, bestPrefix, legend, empty }
     *
     * 与 sparkline 的区别：支持小数刻度（mAP 0.44 这种）、多序列、epoch 横轴，
     * 并标注 best epoch —— 训练曲线专用。
     */
    curve(host, series, opts) {
      opts = opts || {};
      const width = opts.width || 660, height = opts.height || 224;
      const padL = 50, padR = 16, padT = 20, padB = 34;
      const iw = width - padL - padR, ih = height - padT - padB;
      const svg = responsiveSvg(width, height);
      const fmt = opts.fmt || (v => String(v));
      series = (series || []).filter(s => s && s.values && s.values.length);

      const all = [];
      series.forEach(s => s.values.forEach(v => { if (isFinite(v)) all.push(v); }));
      const n = series.length ? Math.max.apply(null, series.map(s => s.values.length)) : 0;

      if (!n || !all.length) {
        // 空状态也把坐标框画出来，否则整块区域会塌成一条线
        svg.appendChild(svgEl('rect', {
          x: padL, y: padT, width: iw, height: ih,
          fill: 'none', stroke: 'var(--line)', 'stroke-width': 1, rx: 4,
        }));
        const t = svgEl('text', {
          x: width / 2, y: height / 2, 'text-anchor': 'middle',
          fill: 'var(--fg-4)', 'font-size': 11.5,
        });
        t.textContent = opts.empty || '暂无训练数据';
        svg.appendChild(t);
        host.innerHTML = '';
        host.appendChild(svg);
        return;
      }

      // 值域：优先用调用方给定范围，否则按数据 ±12% 留白
      let min = opts.yMin != null ? opts.yMin : Math.min.apply(null, all);
      let max = opts.yMax != null ? opts.yMax : Math.max.apply(null, all);
      if (max - min < 1e-9) max = min + Math.max(Math.abs(min) * 0.2, 1e-3);
      const span = max - min;
      if (opts.yMin == null) min -= span * 0.12;
      if (opts.yMax == null) max += span * 0.12;
      if (opts.yFloor0 && min < 0) min = 0;

      const px = i => padL + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw);
      const py = v => padT + ih - ((v - min) / (max - min)) * ih;

      // 横向网格 + Y 轴刻度
      for (let k = 0; k <= 4; k++) {
        const v = min + (max - min) * (k / 4);
        const y = py(v);
        svg.appendChild(svgEl('line', {
          x1: padL, y1: y, x2: width - padR, y2: y, stroke: 'var(--line)', 'stroke-width': 1,
        }));
        const lb = svgEl('text', {
          x: padL - 8, y: y + 3.5, 'text-anchor': 'end',
          fill: 'var(--fg-4)', 'font-size': 9.5, 'font-family': 'var(--font-mono)',
        });
        lb.textContent = fmt(v);
        svg.appendChild(lb);
      }

      // 纵向网格 + X 轴刻度（epoch）
      const xN = Math.max(2, opts.xTicks || 5);
      const xStart = opts.xStart || 1;
      for (let k = 0; k < xN; k++) {
        const i = Math.round((k / (xN - 1)) * (n - 1));
        const x = px(i);
        if (k > 0) {
          svg.appendChild(svgEl('line', {
            x1: x, y1: padT, x2: x, y2: padT + ih,
            stroke: 'var(--line)', 'stroke-width': 1, 'stroke-dasharray': '2 3',
          }));
        }
        const lb = svgEl('text', {
          x: x, y: height - padB + 15,
          'text-anchor': k === 0 ? 'start' : (k === xN - 1 ? 'end' : 'middle'),
          fill: 'var(--fg-4)', 'font-size': 9.5, 'font-family': 'var(--font-mono)',
        });
        lb.textContent = String(xStart + i);
        svg.appendChild(lb);
      }

      // 水平参考线（阈值线）：opts.hlines = [{ v, color, dash, label }]
      // 画在网格之上、折线之下，避免盖住数据。超出当前值域的线自动跳过。
      (opts.hlines || []).forEach(h => {
        if (!isFinite(h.v) || h.v < min || h.v > max) return;
        const y = py(h.v);
        svg.appendChild(svgEl('line', {
          x1: padL, y1: y, x2: width - padR, y2: y,
          stroke: h.color || 'var(--danger)', 'stroke-width': 1,
          'stroke-dasharray': h.dash || '5 4', opacity: 0.75,
        }));
        if (h.label) {
          const t = svgEl('text', {
            x: width - padR - 4, y: y - 4, 'text-anchor': 'end',
            fill: h.color || 'var(--danger)', 'font-size': 9.5,
            'font-family': 'var(--font-mono)', opacity: 0.9,
          });
          t.textContent = h.label;
          svg.appendChild(t);
        }
      });

      // 序列折线
      series.forEach(s => {
        const pts = s.values.map((v, i) => [px(i), py(v)]);
        svg.appendChild(svgEl('path', {
          d: pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' '),
          fill: 'none',
          stroke: s.color || 'var(--accent)',
          'stroke-width': s.width || 1.8,
          'stroke-linejoin': 'round', 'stroke-linecap': 'round',
          'stroke-dasharray': s.dash || '0',
          opacity: s.opacity != null ? s.opacity : 1,
        }));
      });

      // best 峰值标注（取第一条序列）
      if (opts.markBest !== false) {
        const v0 = series[0].values;
        let mi = 0;
        v0.forEach((v, i) => { if (v > v0[mi]) mi = i; });
        svg.appendChild(svgEl('circle', {
          cx: px(mi), cy: py(v0[mi]), r: 3.4,
          fill: 'var(--panel)', stroke: series[0].color || 'var(--accent)', 'stroke-width': 1.8,
        }));
        // bestSuffix 默认 'e'（epoch，训练曲线）；逐帧曲线可传 'f'（frame）等
        const txt = (opts.bestPrefix || 'best ') + fmt(v0[mi]) + ' @'
          + (opts.bestSuffix != null ? opts.bestSuffix : 'e') + (xStart + mi);
        const tw = estTextW(txt, 10);
        const tag = svgEl('text', {
          x: Math.max(padL, Math.min(px(mi) - tw / 2, width - padR - tw)),
          y: Math.max(py(v0[mi]) - 9, padT + 10),
          fill: 'var(--fg-2)', 'font-size': 10, 'font-family': 'var(--font-mono)',
        });
        tag.textContent = txt;
        svg.appendChild(tag);
      }

      // 图例（顶部一行）
      if (opts.legend !== false && series.length) {
        let lx = padL;
        series.forEach(s => {
          const gy = padT - 8;
          svg.appendChild(svgEl('line', {
            x1: lx, y1: gy, x2: lx + 14, y2: gy,
            stroke: s.color || 'var(--accent)', 'stroke-width': 2, 'stroke-dasharray': s.dash || '0',
          }));
          const t = svgEl('text', {
            x: lx + 19, y: gy + 3.5, fill: 'var(--fg-3)', 'font-size': 10,
          });
          t.textContent = s.name || '';
          svg.appendChild(t);
          lx += 19 + estTextW(s.name || '', 10) + 16;
        });
      }

      host.innerHTML = '';
      host.appendChild(svg);
    },

    /** 直方图 */
    histogram(host, bins, opts) {
      opts = opts || {};
      const width = opts.width || 560, height = opts.height || 150;
      const padL = 34, padR = 10, padT = 12, padB = 24;
      const iw = width - padL - padR, ih = height - padT - padB;
      const max = Math.max(1, ...bins.map(b => b.value));
      const svg = responsiveSvg(width, height);

      for (let k = 0; k <= 3; k++) {
        const y = padT + ih * (k / 3);
        svg.appendChild(svgEl('line', {
          x1: padL, y1: y, x2: width - padR, y2: y, stroke: 'var(--line)', 'stroke-width': 1,
        }));
        const lb = svgEl('text', {
          x: padL - 7, y: y + 3.5, 'text-anchor': 'end',
          fill: 'var(--fg-4)', 'font-size': 9.5, 'font-family': 'var(--font-mono)',
        });
        lb.textContent = Math.round(max * (1 - k / 3));
        svg.appendChild(lb);
      }

      const bw = iw / bins.length;
      bins.forEach((b, i) => {
        const h = (b.value / max) * ih;
        const x = padL + i * bw + 1.5;
        // 直方图是有序分箱，用单色渐变比七彩更能读出分布形状
        const t01 = bins.length > 1 ? i / (bins.length - 1) : 1;
        const r = svgEl('rect', {
          x: x, y: padT + ih - h, width: Math.max(1, bw - 3), height: Math.max(0, h),
          rx: 2, fill: 'var(--accent)', opacity: (0.40 + 0.55 * t01).toFixed(2),
        });
        const t = svgEl('title');
        t.textContent = b.label + '：' + b.value;
        r.appendChild(t);
        svg.appendChild(r);
      });

      [0, Math.floor(bins.length / 2), bins.length - 1].forEach(i => {
        if (!bins[i]) return;
        const lb = svgEl('text', {
          x: padL + i * bw + bw / 2, y: height - 7, 'text-anchor': 'middle',
          fill: 'var(--fg-4)', 'font-size': 9.5, 'font-family': 'var(--font-mono)',
        });
        lb.textContent = bins[i].label;
        svg.appendChild(lb);
      });

      host.innerHTML = '';
      host.appendChild(svg);
    },
  };
  BDS.Charts = Charts;

  /* =========================================================== Beidou */
  const Beidou = {
    SYSTEMS: [
      { k: 'BDS', color: '#4d9eff' },
      { k: 'GPS', color: '#3dd68c' },
      { k: 'GAL', color: '#f5a524' },
      { k: 'GLO', color: '#f472b6' },
      { k: 'GNSS', color: '#9aa5b4' },
    ],

    /**
     * NMEA talker ID → 星座标签。
     * 用 `$GNGGA`（多星座合并）时无法从 talker 反推是哪一颗，因此标成 GNSS 而不是
     * 随便认领给 BDS —— 星空图上"颜色认错星座"比"颜色是中性灰"更容易误导人。
     */
    TALKER: { GP: 'GPS', BD: 'BDS', GB: 'BDS', GQ: 'BDS', GL: 'GLO', GA: 'GAL', GN: 'GNSS' },

    /**
     * 把后端 `satellites_detail`（GSV 实测）转成星空图输入。
     * 缺方位角或高度角的条目直接丢弃：没有角度就没有坐标，画不出来。
     * 返回空数组 = 后端没收到 GSV，调用方必须退回到明确标注为"示意"的星座。
     */
    fromDetail(list) {
      const out = [];
      (list || []).forEach((s, i) => {
        if (s.azimuth == null || s.elevation == null) return;
        const tk = String(s.talker || '').toUpperCase();
        out.push({
          prn: s.prn || ('#' + (i + 1)),
          sys: this.TALKER[tk] || 'GNSS',
          az: Math.max(0, Math.min(360, +s.azimuth)),
          el: Math.max(0, Math.min(90, +s.elevation)),
          snr: s.snr == null ? null : +s.snr,
          real: true,
        });
      });
      return out;
    },

    /** 生成确定性的卫星星座（方位角 / 高度角 / 信噪比） */
    constellation(count, seed) {
      const rand = U.rng(seed || 7);
      const list = [];
      for (let i = 0; i < count; i++) {
        const sys = i % 3 === 0 ? 'BDS' : (i % 3 === 1 ? 'GPS' : (i % 5 === 0 ? 'GLO' : 'GAL'));
        list.push({
          prn: sys.slice(0, 1) + String(i + 1).padStart(2, '0'),
          sys: sys,
          az: +(rand() * 360).toFixed(1),
          el: +(8 + rand() * 82).toFixed(1),
          snr: +(30 + rand() * 18).toFixed(1),
        });
      }
      return list;
    },

    /** 极坐标星空图（中心=天顶，外圈=地平线） */
    skyplot(host, sats, opts) {
      opts = opts || {};
      const size = opts.size || 300;
      const c = size / 2, R = c - 30;
      const svg = svgEl('svg', {
        viewBox: '0 0 ' + size + ' ' + size, width: '100%', height: size,
        style: 'max-width:' + size + 'px', class: 'chart-svg',
      });

      // 天区底
      svg.appendChild(svgEl('circle', {
        cx: c, cy: c, r: R, fill: 'var(--panel-2)', stroke: 'var(--line-2)', 'stroke-width': 1,
      }));
      svg.appendChild(svgEl('circle', {
        cx: c, cy: c, r: R * (2 / 3), fill: 'none', stroke: 'var(--line)', 'stroke-width': 1,
        'stroke-dasharray': '3 4',
      }));
      svg.appendChild(svgEl('circle', {
        cx: c, cy: c, r: R / 3, fill: 'none', stroke: 'var(--line)', 'stroke-width': 1,
        'stroke-dasharray': '3 4',
      }));

      // 高度角刻度
      [[0, '0°'], [30, '30°'], [60, '60°']].forEach(t => {
        const r = R * (1 - t[0] / 90);
        const lb = svgEl('text', {
          x: c + 4, y: c - r - 3, fill: 'var(--fg-4)', 'font-size': 9,
          'font-family': 'var(--font-mono)',
        });
        lb.textContent = t[1];
        svg.appendChild(lb);
      });

      // 方位辐条 + 方位标签
      ['N', 'E', 'S', 'W'].forEach((nm, i) => {
        const deg = i * 90;
        const p = polar(c, c, R, deg);
        svg.appendChild(svgEl('line', {
          x1: c, y1: c, x2: p.x, y2: p.y, stroke: 'var(--line)', 'stroke-width': 1,
        }));
        const lp = polar(c, c, R + 15, deg);
        const t = svgEl('text', {
          x: lp.x, y: lp.y + 4, 'text-anchor': 'middle',
          fill: nm === 'N' ? 'var(--accent)' : 'var(--fg-3)',
          'font-size': 11, 'font-weight': 700,
        });
        t.textContent = nm;
        svg.appendChild(t);
      });

      // 扫描线
      const sweep = svgEl('g', { class: 'sky-sweep' });
      const defs = svgEl('defs');
      const rg = svgEl('linearGradient', { id: 'sky-sweep-g', x1: '0%', y1: '0%', x2: '0%', y2: '100%' });
      rg.appendChild(svgEl('stop', { offset: '0%', 'stop-color': 'var(--accent)', 'stop-opacity': 0.5 }));
      rg.appendChild(svgEl('stop', { offset: '100%', 'stop-color': 'var(--accent)', 'stop-opacity': 0 }));
      defs.appendChild(rg);
      sweep.appendChild(defs);
      sweep.appendChild(svgEl('path', {
        d: ['M', c, c, 'L', c - 1, c - R, 'A', R, R, 0, 0, 1, c + 26, c - R * 0.995, 'Z'].join(' '),
        fill: 'url(#sky-sweep-g)',
      }));
      svg.appendChild(sweep);

      // 卫星
      const gs = svgEl('g', { class: 'sky-sats' });
      sats.forEach(s => {
        const r = R * (1 - s.el / 90);
        const p = polar(c, c, Math.max(6, r), s.az);
        const sys = this.SYSTEMS.find(x => x.k === s.sys) || this.SYSTEMS[0];
        const g = svgEl('g', { class: 'sky-sat' });
        g.appendChild(svgEl('circle', { cx: p.x, cy: p.y, r: 8, fill: sys.color, opacity: 0.14 }));
        g.appendChild(svgEl('circle', { cx: p.x, cy: p.y, r: 3.6, fill: sys.color }));
        const t = svgEl('title');
        t.textContent = s.prn + ' · ' + s.sys + '\n方位角 ' + s.az + '°  高度角 ' + s.el + '°\n信噪比 ' + s.snr + ' dBHz';
        g.appendChild(t);
        if (s.el > 34) {
          const lb = svgEl('text', {
            x: p.x, y: p.y - 8, 'text-anchor': 'middle',
            fill: 'var(--fg-3)', 'font-size': 8.5, 'font-family': 'var(--font-mono)',
          });
          lb.textContent = s.prn;
          g.appendChild(lb);
        }
        gs.appendChild(g);
      });
      svg.appendChild(gs);

      // 天顶标记
      svg.appendChild(svgEl('circle', { cx: c, cy: c, r: 2, fill: 'var(--fg-4)' }));

      // 空数据时给出明确说明。宁可画一张空图，也不要画一张"看起来有卫星"的假图。
      if (!sats.length && opts.empty) {
        const t = svgEl('text', {
          x: c, y: c + 3, 'text-anchor': 'middle',
          fill: 'var(--fg-4)', 'font-size': 11,
        });
        t.textContent = opts.empty;
        svg.appendChild(t);
      }

      host.innerHTML = '';
      host.appendChild(svg);
    },

    /**
     * ENU 目标散布图。
     *
     * ``opts.track``  —— 轨迹折线 ``[{e,n}]``（设备走过的路线），画在目标点之下；
     * ``opts.marker`` —— 当前时刻的位置标记 ``{e,n}``，拖动时间轴时跟着动。
     *
     * 两者与目标点**共用同一个 ENU 基准与比例尺**：基准取当前定位帧，
     * 目标点由像素偏移换算、轨迹由经纬度换算，用的是同一组常数。
     */
    scatter(host, points, opts) {
      opts = opts || {};
      const width = opts.width || 520, height = opts.height || 300;
      const pad = 38;
      const iw = width - pad * 2, ih = height - pad * 2;
      const svg = responsiveSvg(width, height);

      const track = opts.track || [];
      const marker = opts.marker || null;
      const hasData = points.length > 0 || track.length > 0 || !!marker;
      let maxE = 50, maxN = 50;
      if (hasData) {
        maxE = 1; maxN = 1;
        points.forEach(p => {
          maxE = Math.max(maxE, Math.abs(p.e));
          maxN = Math.max(maxN, Math.abs(p.n));
        });
        track.forEach(p => {
          maxE = Math.max(maxE, Math.abs(p.e));
          maxN = Math.max(maxN, Math.abs(p.n));
        });
        if (marker) {
          maxE = Math.max(maxE, Math.abs(marker.e));
          maxN = Math.max(maxN, Math.abs(marker.n));
        }
      }
      const scale = Math.min(iw / 2 / maxE, ih / 2 / maxN) * 0.92;
      const cx = width / 2, cy = height / 2;
      const toX = e => cx + e * scale;
      const toY = n => cy - n * scale;

      // 栅格 + 刻度（无数据时不画，避免出现无意义的 0/1 刻度）
      if (hasData) {
        const stepM = niceStep(Math.max(maxE, maxN) / 4);
        const span = Math.max(maxE, maxN);
        const mono = 'var(--font-mono)';
        const fmtT = stepM < 1 ? v => v.toFixed(1) : v => String(Math.round(v));
        for (let m = -Math.ceil(span / stepM) * stepM; m <= span + 1e-6; m += stepM) {
          const x = cx + m * scale, y = cy - m * scale;
          const near = Math.abs(m) < 1e-6;
          if (x > pad - 4 && x < width - pad + 4) {
            svg.appendChild(svgEl('line', {
              x1: x, y1: pad - 6, x2: x, y2: height - pad + 6, stroke: 'var(--line)', 'stroke-width': 1,
            }));
            if (!near) {
              const lb = svgEl('text', {
                x: x, y: height - pad + 18, 'text-anchor': 'middle',
                fill: 'var(--fg-4)', 'font-size': 9, 'font-family': mono,
              });
              lb.textContent = fmtT(m);
              svg.appendChild(lb);
            }
          }
          if (y > pad - 4 && y < height - pad + 4) {
            svg.appendChild(svgEl('line', {
              x1: pad - 6, y1: y, x2: width - pad + 6, y2: y, stroke: 'var(--line)', 'stroke-width': 1,
            }));
            if (!near) {
              const lb = svgEl('text', {
                x: pad - 10, y: y + 3.5, 'text-anchor': 'end',
                fill: 'var(--fg-4)', 'font-size': 9, 'font-family': mono,
              });
              lb.textContent = fmtT(m);
              svg.appendChild(lb);
            }
          }
        }
      }

      // 坐标轴
      svg.appendChild(svgEl('line', { x1: pad - 8, y1: cy, x2: width - pad + 8, y2: cy, stroke: 'var(--line-2)', 'stroke-width': 1.2 }));
      svg.appendChild(svgEl('line', { x1: cx, y1: pad - 8, x2: cx, y2: height - pad + 8, stroke: 'var(--line-2)', 'stroke-width': 1.2 }));

      const ax = svgEl('text', { x: width - pad, y: cy - 7, 'text-anchor': 'end', fill: 'var(--fg-4)', 'font-size': 9.5, 'font-family': 'var(--font-mono)' });
      ax.textContent = 'E →';
      svg.appendChild(ax);
      const ay = svgEl('text', { x: cx + 6, y: pad - 12, fill: 'var(--fg-4)', 'font-size': 9.5, 'font-family': 'var(--font-mono)' });
      ay.textContent = '↑ N';
      svg.appendChild(ay);

      // 原点（当前定位帧的位置）
      svg.appendChild(svgEl('circle', { cx: cx, cy: cy, r: 9, fill: 'var(--geo)', opacity: 0.16 }));
      svg.appendChild(svgEl('circle', { cx: cx, cy: cy, r: 3, fill: 'var(--geo)' }));

      // 轨迹折线（设备走过的路线）
      if (track.length > 1) {
        svg.appendChild(svgEl('polyline', {
          points: track.map(p => toX(p.e) + ',' + toY(p.n)).join(' '),
          fill: 'none', stroke: 'var(--geo)', 'stroke-width': 1.6,
          'stroke-opacity': 0.55, 'stroke-linejoin': 'round',
        }));
        // 起点单独标出来：拖动时才知道"从哪出发"
        svg.appendChild(svgEl('circle', {
          cx: toX(track[0].e), cy: toY(track[0].n), r: 3,
          fill: 'none', stroke: 'var(--geo)', 'stroke-width': 1.4,
        }));
      }

      // 目标点
      points.forEach(p => {
        const x = toX(p.e), y = toY(p.n);
        const g = svgEl('g');
        g.appendChild(svgEl('circle', { cx: x, cy: y, r: 6.5, fill: p.color, opacity: 0.16 }));
        g.appendChild(svgEl('circle', { cx: x, cy: y, r: 3, fill: p.color }));
        const t = svgEl('title');
        t.textContent = p.label + '\n东 ' + p.e + ' m  北 ' + p.n + ' m';
        g.appendChild(t);
        svg.appendChild(g);
      });

      // 当前时刻的位置十字（拖动时间轴时它跟着走）
      if (marker) {
        const x = toX(marker.e), y = toY(marker.n);
        const g = svgEl('g');
        g.appendChild(svgEl('circle', { cx: x, cy: y, r: 8, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 1.6 }));
        g.appendChild(svgEl('line', { x1: x - 12, y1: y, x2: x + 12, y2: y, stroke: 'var(--accent)', 'stroke-width': 1 }));
        g.appendChild(svgEl('line', { x1: x, y1: y - 12, x2: x, y2: y + 12, stroke: 'var(--accent)', 'stroke-width': 1 }));
        const t = svgEl('title');
        t.textContent = marker.label || '当前位置';
        g.appendChild(t);
        svg.appendChild(g);
      }

      if (!hasData) {
        const t = svgEl('text', { x: width / 2, y: height / 2, 'text-anchor': 'middle', fill: 'var(--fg-4)', 'font-size': 11.5 });
        t.textContent = opts.emptyText || '暂无目标定位数据';
        svg.appendChild(t);
      }

      host.innerHTML = '';
      host.appendChild(svg);
    },
  };

  /** 估算文本像素宽度（CJK 按 1em、其余按 0.56em），用于图例排布与标签防溢出 */
  function estTextW(s, fs) {
    let w = 0;
    for (let i = 0; i < s.length; i++) {
      w += s.charCodeAt(i) > 0x2e80 ? fs : fs * 0.56;
    }
    return w;
  }

  function niceStep(v) {
    const p = Math.pow(10, Math.floor(Math.log10(Math.max(1e-6, v))));
    const n = v / p;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * p;
  }

  BDS.Beidou = Beidou;

})(window);

function make_figures(csvPath, outDir)
%MAKE_FIGURES  从 ultralytics 的 results.csv 重绘训练曲线（MATLAB 版）。
%
% 为什么不直接用 run 目录里现成的 results.png：
%   那张图是 ultralytics 在训练结束时画的，而页面上的 KPI 是另一套口径 ——
%   KPI 取「mAP50 峰值」那一轮（epoch 74），best.pt 却按 fitness
%   （0.1*mAP50 + 0.9*mAP50-95）取「第 66 轮」。两者差 0.4 个百分点，
%   同屏出现就像数据是拼凑的。这里直接读 results.csv 重画，
%   与页面曲线、KPI 同源，口径不会再打架。
%
% 用法：
%   make_figures('<run>\results.csv', '<project>\修正版代码\figures')

if ~exist(outDir, 'dir'); mkdir(outDir); end
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultFigureColor', 'w');

T = readtable(csvPath, 'VariableNamingRule', 'preserve');

ep       = T.('epoch');
mAP50    = T.('metrics/mAP50(B)');
mAP50_95 = T.('metrics/mAP50-95(B)');
prec     = T.('metrics/precision(B)');
rec      = T.('metrics/recall(B)');
tr_box   = T.('train/box_loss');
tr_cls   = T.('train/cls_loss');
tr_dfl   = T.('train/dfl_loss');
va_box   = T.('val/box_loss');
va_cls   = T.('val/cls_loss');
va_dfl   = T.('val/dfl_loss');
lr0      = T.('lr/pg0');

% ---- 两个「最优」不是同一轮：KPI 用前者，best.pt 用后者 ----
[v_map, i_map] = max(mAP50);
fit = 0.1 * mAP50 + 0.9 * mAP50_95;
[~, i_fit] = max(fit);
e_map = ep(i_map);
e_fit = ep(i_fit);

% ---- 配色（色盲友好，与网页强调色同系）----
C_MAP   = [0.180 0.435 0.851];
C_MAP95 = [0.122 0.616 0.333];
C_P     = [0.851 0.510 0.169];
C_R     = [0.753 0.224 0.169];
C_GRID  = [0.85 0.86 0.88];

% =====================================================================
% 1. mAP 收敛曲线（论文主图）
% =====================================================================
f = figure('Units', 'centimeters', 'Position', [2 2 16.5 10.4]);
hold on
p1 = plot(ep, mAP50,    '-', 'Color', C_MAP,   'LineWidth', 1.8);
p2 = plot(ep, mAP50_95, '-', 'Color', C_MAP95, 'LineWidth', 1.8);
p3 = plot(e_map, v_map, 'o', 'MarkerSize', 8, 'LineWidth', 1.2, ...
          'MarkerFaceColor', C_MAP, 'MarkerEdgeColor', 'w');
p4 = plot(e_fit, mAP50(i_fit), 's', 'MarkerSize', 8, 'LineWidth', 1.2, ...
          'MarkerFaceColor', C_MAP95, 'MarkerEdgeColor', 'w');
style_ax(gca);
xlim([0, max(ep) + 2]); ylim([0, 0.52]);
xlabel('Epoch'); ylabel('Metric');
title('Detection accuracy vs. epoch (VisDrone2019-DET val)');
room_for_title();
legend([p1 p2 p3 p4], { ...
    'mAP@0.5', 'mAP@0.5:0.95', ...
    sprintf('mAP@0.5 peak: %.4f @ epoch %d', v_map, e_map), ...
    sprintf('best.pt (fitness): %.4f @ epoch %d', mAP50(i_fit), e_fit)}, ...
    'Location', 'southeast', 'Box', 'off', 'FontSize', 8.5);
text(max(ep) * 0.02, 0.47, sprintf('%d epochs · 10 classes · 17,552,134 params', numel(ep)), ...
    'FontSize', 8.5, 'Color', [0.35 0.38 0.42]);
save_fig(f, outDir, 'matlab_map.png');

% =====================================================================
% 2. 精确率 / 召回率
% =====================================================================
f = figure('Units', 'centimeters', 'Position', [2 2 16.5 10.4]);
hold on
plot(ep, prec, '-',  'Color', C_P, 'LineWidth', 1.8);
plot(ep, rec,  '--', 'Color', C_R, 'LineWidth', 1.8);
style_ax(gca);
xlim([0, max(ep) + 2]); ylim([0, 0.75]);
xlabel('Epoch'); ylabel('Metric');
title('Precision / Recall vs. epoch');
room_for_title();
legend({'Precision', 'Recall'}, 'Location', 'southeast', 'Box', 'off', 'FontSize', 8.5);
save_fig(f, outDir, 'matlab_pr.png');

% =====================================================================
% 3. 损失：训练集 vs 验证集
% =====================================================================
f = figure('Units', 'centimeters', 'Position', [2 2 24 8]);
tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
names = {'box', 'cls', 'dfl'};
trs   = {tr_box, tr_cls, tr_dfl};
vas   = {va_box, va_cls, va_dfl};
for k = 1:3
    nexttile; hold on
    plot(ep, trs{k}, '-',  'Color', C_MAP, 'LineWidth', 1.6);
    plot(ep, vas{k}, '--', 'Color', C_R,   'LineWidth', 1.6);
    style_ax(gca);
    xlim([0, max(ep) + 2]);
    xlabel('Epoch'); ylabel('Loss');
    title([names{k} ' loss']);
    if k == 1
        legend({'train', 'val'}, 'Location', 'northeast', 'Box', 'off', 'FontSize', 8.5);
    end
end
title(tl, 'Training / validation loss', 'FontWeight', 'bold');
save_fig(f, outDir, 'matlab_loss.png');

% =====================================================================
% 4. 学习率调度
% =====================================================================
f = figure('Units', 'centimeters', 'Position', [2 2 16.5 7.5]);
hold on
plot(ep, lr0, '-', 'Color', C_MAP, 'LineWidth', 1.8);
style_ax(gca);
xlim([0, max(ep) + 2]);
xlabel('Epoch'); ylabel('lr (pg0)');
title('Learning-rate schedule (AdamW, cosine)');
room_for_title();
save_fig(f, outDir, 'matlab_lr.png');

% =====================================================================
% 5. 2x5 全指标面板（布局对齐 ultralytics 的 results.png，便于替换）
% =====================================================================
f = figure('Units', 'centimeters', 'Position', [2 2 30 11.5]);
tl = tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
panels = { ...
    'train/box\_loss', tr_box,   C_MAP; ...
    'train/cls\_loss', tr_cls,   C_MAP; ...
    'train/dfl\_loss', tr_dfl,   C_MAP; ...
    'precision(B)',    prec,     C_P;   ...
    'recall(B)',       rec,      C_R;   ...
    'val/box\_loss',   va_box,   C_MAP95; ...
    'val/cls\_loss',   va_cls,   C_MAP95; ...
    'val/dfl\_loss',   va_dfl,   C_MAP95; ...
    'mAP50(B)',        mAP50,    C_MAP; ...
    'mAP50-95(B)',     mAP50_95, C_MAP95};
for k = 1:size(panels, 1)
    nexttile; hold on
    y = panels{k, 2};
    plot(ep, y, '-', 'Color', panels{k, 3}, 'LineWidth', 1.5);
    style_ax(gca);
    xlim([0, max(ep) + 2]);
    title(panels{k, 1}, 'FontSize', 9);
    if k >= 9
        [vv, ii] = max(y);
        plot(ep(ii), vv, 'o', 'MarkerSize', 5, 'LineWidth', 1, ...
             'MarkerFaceColor', panels{k, 3}, 'MarkerEdgeColor', 'w');
    end
    if k > 5; xlabel('Epoch', 'FontSize', 8.5); end
end
save_fig(f, outDir, 'matlab_panel.png');

% =====================================================================
fprintf('OK csv=%s\n', csvPath);
fprintf('epochs=%d\n', numel(ep));
fprintf('mAP50_peak=%.5f@epoch%d\n', v_map, e_map);
fprintf('fitness_best=%.5f@epoch%d (mAP50=%.5f)\n', fit(i_fit), e_fit, mAP50(i_fit));
fprintf('out=%s\n', outDir);
end

% ---------------------------------------------------------------------
function style_ax(ax)
grid(ax, 'on');
set(ax, 'GridAlpha', 0.22, 'GridColor', [0.75 0.77 0.80], ...
        'Box', 'on', 'LineWidth', 0.7, 'FontSize', 9, ...
        'TickDir', 'out', 'XColor', [0.28 0.31 0.35], 'YColor', [0.28 0.31 0.35]);
end

function save_fig(f, outDir, name)
% Padding='figure' 保留整张画布，不要 tight 裁剪。
% 默认的 tight 裁剪会把标题顶端切掉一行 —— exportgraphics 不报错，
% 只是悄悄少几行像素。
p = fullfile(outDir, name);
exportgraphics(f, p, 'Resolution', 200, 'BackgroundColor', 'white', 'Padding', 'figure');
close(f);
fprintf('WROTE %s\n', p);
end

function room_for_title()
% 给标题留出顶部余量。
% exportgraphics 的 tight 裁剪会切标题，改成 Padding='figure' 后不切了，
% 但坐标区上边框会紧贴标题。把坐标区高度压 7%（底边不动）即可。
% 只在独立 figure 里用；tiledlayout 的坐标区位置由布局管理器管，别手动改。
ax = gca;
pos = ax.Position;
ax.Position = [pos(1), pos(2), pos(3), pos(4) * 0.93];
end

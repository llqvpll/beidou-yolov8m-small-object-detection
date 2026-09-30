@echo off
chcp 65001 >nul
cd /d %~dp0
title 生成论文配图 - MATLAB 曲线 + 重跑验证

set "PYEXE=C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
set "PYTHONPATH="
set "POLARS_SKIP_CPU_CHECK=1"

if not exist "%PYEXE%" goto nopy

echo ==========================================================
echo   生成论文配图
echo.
echo   把「分析 · 模型与训练」那一块配图重做成与当前权重
echo   严格对应的一批，输出到 修正版代码\figures\
echo.
echo   MATLAB 画:   mAP / P / R / 损失 / 学习率（读 results.csv）
echo   重跑验证:    PR / F1 / P-R 曲线 + 混淆矩阵（用当前权重）
echo.
echo   耗时: MATLAB 约 1 分钟 + 验证约 1 分钟
echo   页面上的 44.6%% 与 44.2%% 是两个不同的「最优」口径，
echo   脚本结束时会把两者都打出来。
echo ==========================================================
echo.

"%PYEXE%" -u 生成论文配图.py %*
set RC=%ERRORLEVEL%
echo.
echo -------- 退出码 %RC% --------
if not "%RC%"=="0" goto failed

echo 完成。回到系统页面，点「分析 · 模型与训练 · 重读」即可看到新图。
pause
exit /b 0

:failed
echo.
echo 出错了。常见原因：
echo   1. 找不到 MATLAB      -^> 只重跑验证:  生成论文配图.bat --no-matlab
echo   2. 找不到 run 目录    -^> 指定:        生成论文配图.bat --project C:/yolo_runs/train --run visdrone_v4
echo   3. 显存不够           -^> 降 batch:    生成论文配图.bat --batch 2
echo   4. 数据集路径不对     -^> 指定:        生成论文配图.bat --data "%~dp0VisDrone.yaml"
pause
exit /b %RC%

:nopy
echo 找不到 Python: %PYEXE%
echo 请改本文件里的 PYEXE 路径后再试。
pause
exit /b 1

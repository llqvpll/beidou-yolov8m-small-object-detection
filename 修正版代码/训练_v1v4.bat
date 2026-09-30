@echo off
chcp 65001 >nul
cd /d %~dp0
title VisDrone v1+v4 训练 - imgsz640 batch4 150ep

set "PYEXE=C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
set "PYTHONPATH="
set "POLARS_SKIP_CPU_CHECK=1"

if not exist "%PYEXE%" goto nopy

echo ==========================================================
echo   v1 基线 + v4 推荐    顺序训练
echo   imgsz 640 / batch 4 / 150 epoch
echo   输出目录: C:\yolo_runs\train
echo.
echo   已跑满的变体会自动跳过
echo   半途中断会打印续训命令, 不会静默从头再来
echo ==========================================================
echo.

"%PYEXE%" -u 连续训练_v1v4.py --imgsz 640 --batch 4 --epochs 150 --project C:/yolo_runs/train
set RC=%ERRORLEVEL%
echo.
echo -------- 退出码 %RC% --------
if not "%RC%"=="0" goto failed

echo 全部完成。
echo 下一步出对比图: "%PYEXE%" 对比指标_v1v4.py --project C:/yolo_runs/train
pause
exit /b 0

:failed
echo 出错了。先看这两个日志:
echo   C:\yolo_runs\train\visdrone_v1.log
echo   C:\yolo_runs\train\visdrone_v4.log
echo.
echo 常见原因: 显存不够。降一档重跑即可, 例如 imgsz 640 -^> 512, 或 batch 4 -^> 3。
pause
exit /b %RC%

:nopy
echo 找不到 Python: %PYEXE%
echo 请改本文件里的 PYEXE 路径后再试。
pause
exit /b 1

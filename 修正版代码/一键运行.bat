@echo off
setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title VisDrone + YOLOv8m one click run

rem polars(ultralytics dependency) CPU check false-positives on this machine
set "POLARS_SKIP_CPU_CHECK=1"

echo ============================================================
echo   VisDrone + YOLOv8m improved detector -- one click run
echo   work dir: %CD%
echo ============================================================
echo.

rem ==================== 1. locate a working python ====================
set "PYEXE="
for /f "delims=" %%i in ('where python 2^>nul') do if not defined PYEXE set "PYEXE=%%i"
if defined PYEXE (
  "%PYEXE%" -c "import sys" >nul 2>&1 || set "PYEXE="
)
if not defined PYEXE (
  for %%D in (Python313 Python312 Python311 Python310 Python39) do (
    if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\%%D\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
  )
)
if not defined PYEXE (
  echo [ERROR] Python not found.
  echo         Install Python 3.9+ from https://www.python.org/downloads/
  echo         and tick "Add python.exe to PATH", then run this file again.
  echo.
  pause
  exit /b 1
)
echo [1/6] Python: %PYEXE%
"%PYEXE%" -c "import sys; print('        version', sys.version.split()[0])"

rem ==================== 2. torch + ultralytics ====================
echo.
echo [2/6] checking torch ...

rem GPU probe: ask nvidia-smi for the actual adapter name. The mere presence
rem of the exe proves nothing - leftover driver stubs are common on desktops.
rem On Windows the default PyPI torch is CPU-only, and a CPU-only or pre-sm_120
rem torch silently falls back to CPU training, so verify and self-heal here.
rem RTX 50 series (Blackwell, sm_120) needs a torch built with CUDA 12.8 (cu128).
set "GPU_NAME="
for /f "delims=" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do if not defined GPU_NAME set "GPU_NAME=%%g"

"%PYEXE%" -c "import torch" >nul 2>&1
if errorlevel 1 (
  if defined GPU_NAME (
    echo        torch missing, GPU found: %GPU_NAME%
    echo        installing CUDA 12.8 build - RTX 50 series needs cu128 for sm_120 ...
    "%PYEXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ) else (
    echo        no torch yet and no NVIDIA GPU - installing CPU-only build ...
    "%PYEXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  )
  if errorlevel 1 echo        [WARN] torch install failed - training will likely fail.
) else if defined GPU_NAME (
  "%PYEXE%" -c "import torch;raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
  if errorlevel 1 (
    echo        [FIX] GPU found ^(%GPU_NAME%^) but torch cannot use it - the installed
    echo        build is CPU-only or too old for sm_120. Replacing with CUDA 12.8 build ...
    "%PYEXE%" -m pip uninstall -y torch torchvision torchaudio
    "%PYEXE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  )
)
"%PYEXE%" -c "import torch; print('        torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

"%PYEXE%" -c "import ultralytics" >nul 2>&1
if errorlevel 1 (
  echo        ultralytics not installed - installing now, may take several minutes ...
  "%PYEXE%" -m pip install -U ultralytics
  if errorlevel 1 (
    echo [ERROR] pip install ultralytics failed. Check your network / proxy.
    pause
    exit /b 1
  )
)
"%PYEXE%" -c "import ultralytics; print('        ultralytics', ultralytics.__version__)"
echo        patching target: %PYEXE%

rem remember GPU availability for the training menu below
rem write to a temp file instead of for /f - nested quotes inside for /f are
rem fragile under cmd, while a plain redirect plus set /p is bullet-proof
set "CUDA_OK=0"
"%PYEXE%" -c "import torch;print(1 if torch.cuda.is_available() else 0)" > "%TEMP%\visdrone_cuda_flag.txt" 2>nul
set /p CUDA_OK=<"%TEMP%\visdrone_cuda_flag.txt" 2>nul
if not defined CUDA_OK set "CUDA_OK=0"

rem GPU health check: RTX 50 series is Blackwell sm_120 and needs a torch built
rem with CUDA 12.8 or newer. torch.cuda.is_available alone does NOT prove it can
rem train - the failure only shows up at the first kernel launch. So actually run one.
if "%CUDA_OK%"=="1" (
  echo.
  echo        GPU detected - running health check ...
  "%PYEXE%" 显卡检查.py
  if errorlevel 1 (
    echo.
    echo   [WARN] GPU check failed - read the hints above before training.
    echo          Most common cause: RTX 50 series needs a torch built with CUDA 12.8+.
    echo          Fix:  pip uninstall -y torch torchvision
    echo                pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  )
)

rem ==================== 3. pretrained weights ====================
echo.
echo [3/6] checking yolov8m.pt ...
if exist "yolov8m.pt" (
  echo        found: %CD%\yolov8m.pt
) else (
  echo        missing - downloading ~50MB ...
  set "W1=https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt"
  set "W2=https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt"
  set "W3=https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt"

  rem --ssl-no-revoke: some networks cannot reach the CRL/OCSP server,
  rem schannel then fails with CRYPT_E_NO_REVOCATION_CHECK - curl error 35.
  curl -L --ssl-no-revoke --connect-timeout 20 -o yolov8m.pt "%W1%"
  if not exist "yolov8m.pt" (
    echo        retry via mirror 1 ...
    curl -L --ssl-no-revoke --connect-timeout 20 -o yolov8m.pt "%W2%"
  )
  if not exist "yolov8m.pt" (
    echo        retry via mirror 2 ...
    curl -L --ssl-no-revoke --connect-timeout 20 -o yolov8m.pt "%W3%"
  )
  if not exist "yolov8m.pt" (
    echo.
    echo [WARN] could not download yolov8m.pt automatically.
    echo        Please download it manually and put it in: %CD%
    echo        URL: %W1%
    echo.
  )
)

rem ==================== 4. install custom modules ====================
echo.
echo [4/6] installing custom modules into ultralytics ...
"%PYEXE%" install_custom_modules.py
if errorlevel 1 (
  echo [ERROR] install_custom_modules.py failed.
  pause
  exit /b 1
)

rem ==================== 5. self test ====================
echo.
echo [5/6] self test ...
"%PYEXE%" test_modules.py

rem ==================== 6. train ====================
echo.
echo [6/6] pick a variant to train:
echo        v1 = baseline                DualConv, 4 scales
echo        v2 = Slim-Neck               4 scales
echo        v3 = prune P5                3 scales
echo        v4 = Slim-Neck + prune P5    RECOMMENDED
echo        s  = smoke test (synthetic mini dataset, 3 epochs, CPU)
echo        c  = environment check  python 环境验证.py
echo        q  = quit without training
echo.
if "%CUDA_OK%"=="0" (
  echo   ------------------------------------------------------------
  echo   [NOTICE] No NVIDIA GPU on this machine.
  echo            v1~v4 are real training runs - they need a GPU.
  echo            On CPU a full VisDrone run would take weeks, so it is blocked.
  echo            - to verify the code path now: press  s
  echo            - to get real results: run on a free GPU platform - see KAGGLE.md
  echo   ------------------------------------------------------------
)
if "%CUDA_OK%"=="1" (
  echo   ------------------------------------------------------------
  echo   GPU detected - v1~v4 will train on it.
  echo   batch defaults to -1: train_v8.py profiles real forward+backward
  echo   peak VRAM and picks the largest safe batch ^(ultralytics AutoBatch
  echo   is unreliable on Windows - WDDM shared memory inflates the number^).
  echo   If the picked batch is 1 or 2, rerun with --imgsz 800 -
  echo   but keep the SAME imgsz for all four variants or the ablation dies.
  echo   ------------------------------------------------------------
)
echo   note: full training needs the VisDrone2019-DET dataset.
echo         edit "path:" in VisDrone.yaml to point at it first.
echo         option "s" needs no dataset - it uses 冒烟测试数据集 automatically.
echo   tip: press c for a full environment check - GPU, deps, patch, weights, dataset.
echo.
:pick
set "V="
set /p "V=variant, default v4: "
if /i "%V%"=="q" goto :done
if "%V%"=="" set "V=v4"
if /i "%V%"=="s" goto :smoke
if /i "%V%"=="c" goto :envcheck
if /i "%V%"=="v1" goto :real
if /i "%V%"=="v2" goto :real
if /i "%V%"=="v3" goto :real
if /i "%V%"=="v4" goto :real
echo        invalid choice "%V%" - please type v1 / v2 / v3 / v4 / s / c / q.
goto :pick

:envcheck
echo.
"%PYEXE%" 环境验证.py
echo.
goto :pick

:real
if "%CUDA_OK%"=="0" (
  echo.
  echo        [BLOCKED] no GPU - real training is not possible on this machine.
  echo                  use  s  for the smoke test, or train on Kaggle - see KAGGLE.md.
  echo                  if you really must use CPU, run it yourself:
  echo                    python train_v8.py --variant %V% --force-cpu
  echo.
  goto :pick
)
echo.
echo        training variant %V% ...  press Ctrl+C to abort
echo.
"%PYEXE%" train_v8.py --variant %V% --name visdrone_%V%
set "RC=%ERRORLEVEL%"
if "%RC%"=="3" (
  echo.
  echo [NOTE] training was skipped - see the message above: no usable GPU.
  goto :done
)
if not "%RC%"=="0" (
  echo.
  echo [ERROR] training failed. Please read the traceback above.
  pause
  exit /b 1
)
set "RAN=1"
goto :done

:smoke
echo.
echo        building synthetic mini dataset ...
"%PYEXE%" make_smoke_data.py
echo.
echo        smoke test ...  press Ctrl+C to abort
echo        This only proves the pipeline works; real accuracy needs a GPU + VisDrone.
echo.
"%PYEXE%" train_v8.py --variant v4 --quick
if errorlevel 1 (
  echo.
  echo [ERROR] smoke test failed. Please read the traceback above.
  pause
  exit /b 1
)
set "V=smoke_v4"
set "RAN=1"

:done
rem ultralytics auto-suffixes the run folder (-2, -3 ...) when the name is
rem taken, so never hardcode it - the newest folder under runs\train is
rem always the one this invocation just produced
set "RUN_DIR="
if exist "%CD%\runs\train" for /f "delims=" %%d in ('dir /b /ad /o-d "%CD%\runs\train" 2^>nul') do if not defined RUN_DIR set "RUN_DIR=%%d"
echo.
echo ============================================================
echo   finished.
if defined RUN_DIR echo   results:     %CD%\runs\train\%RUN_DIR%\
if not "%RAN%"=="1" goto :banner_end
if /i "%V%"=="smoke_v4" (
  echo   note: smoke weights only prove the pipeline runs - nothing to validate.
) else (
  echo   validate:    yolo val model="%CD%\runs\train\%RUN_DIR%\weights\best.pt" data=VisDrone.yaml
)
:banner_end
echo ============================================================
pause
endlocal

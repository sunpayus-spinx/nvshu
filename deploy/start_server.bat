@echo off
REM NuShuPARS Web 服务启动（Windows / 老笔记本）
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"

set PYTHONNOUSERSITE=1
set KMP_DUPLICATE_LIB_OK=1
set YOLO_CONFIG_DIR=%ROOT%\models_cache\ultralytics
set NVSHU_REC_MODEL_DIR=%ROOT%\trocr_model_nushu
set NVSHU_CHARCLS_CKPT=%ROOT%\models\nushu_unicode_best.pt

REM 用系统 python（conda/anaconda 亦可，环境变量已兼容）
set PY=python
where %PY% >nul 2>nul || set PY=py -3

echo [1/3] starting uvicorn on 0.0.0.0:8000 ...
start "nvshu-server" /min %PY% -m uvicorn server:app --host 0.0.0.0 --port 8000

echo [2/3] waiting for /health ...
%PY% deploy\post_start_tune.py http://127.0.0.1:8000

echo [3/3] smoke test ...
%PY% deploy\smoke_test.py http://127.0.0.1:8000

echo.
echo Done. Frontend: http://127.0.0.1:8000/web
endlocal

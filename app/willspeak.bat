@echo off
setlocal
set "PY="
set "SCRIPT="
cd /d "%~dp0"

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PY=%VIRTUAL_ENV%\Scripts\python.exe"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=%CD%\.venv\Scripts\python.exe"
if not defined PY if exist "..\.venv\Scripts\python.exe" set "PY=%CD%\..\.venv\Scripts\python.exe"
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 set "PY=python"
)

if "%~1"=="setup" goto setup
if "%~1"=="train-setup" goto train_setup
if not defined PY (
  echo Python was not found. Run "willspeak setup" first.
  exit /b 1
)
if "%~1"=="" goto menu
set "CMD=%~1"
shift
goto dispatch

:menu
echo willspeak
echo   0^) setup        first-time install (.venv, requirements, Parakeet)
echo   1^) run          start the server (then open http://127.0.0.1:8000/live)
echo   2^) check        load the model, decode one clip
echo   3^) fetch-models download Parakeet
echo   4^) bench        speed benchmark on this box
echo   5^) evaluate     WER on an export
echo   6^) train        fine-tune (training machine)
echo   7^) export       fine-tune -^> int8 ONNX
echo   8^) torgo        fetch TORGO ("torgo manifest" builds manifests)
echo   9^) match-eval   matcher evaluation
echo  10^) test         run the tests
echo   q^) quit
set /p choice="> "
if "%choice%"=="0" goto setup
if "%choice%"=="1" set "CMD=run"
if "%choice%"=="2" set "CMD=check"
if "%choice%"=="3" set "CMD=fetch-models"
if "%choice%"=="4" set "CMD=bench"
if "%choice%"=="5" set "CMD=evaluate"
if "%choice%"=="6" set "CMD=train"
if "%choice%"=="7" set "CMD=export"
if "%choice%"=="8" set "CMD=torgo"
if "%choice%"=="9" set "CMD=match-eval"
if "%choice%"=="10" set "CMD=test"
if not defined CMD exit /b 0

:dispatch
if "%CMD%"=="help" goto usage
if "%CMD%"=="--help" goto usage
if "%CMD%"=="-h" goto usage
if "%CMD%"=="run" set "SCRIPT=run.py"
if "%CMD%"=="check" set "SCRIPT=scripts\asr\check.py"
if "%CMD%"=="fetch-models" set "SCRIPT=scripts\fetch_models.py"
if "%CMD%"=="bench" set "SCRIPT=scripts\asr\bench.py"
if "%CMD%"=="evaluate" set "SCRIPT=scripts\asr\evaluate.py"
if "%CMD%"=="train" set "SCRIPT=scripts\asr\train.py"
if "%CMD%"=="export" set "SCRIPT=scripts\asr\export.py"
if "%CMD%"=="lab" set "SCRIPT=lab.py"
if "%CMD%"=="match-eval" set "SCRIPT=scripts\matcher\evaluate.py"
if "%CMD%"=="test" goto tests
if "%CMD%"=="torgo" goto torgo
if defined SCRIPT goto arguments
echo unknown command: %CMD% 1>&2
exit /b 2

:torgo
set "SCRIPT=scripts\corpus\fetch_torgo.py"
if not "%~1"=="manifest" goto arguments
set "SCRIPT=scripts\corpus\torgo_manifest.py"
shift

:arguments
set "ARGS="
:next_argument
if "%~1"=="" goto invoke
set "ARGS=%ARGS% %1"
shift
goto next_argument

:invoke
"%PY%" "%SCRIPT%" %ARGS%
exit /b %errorlevel%

:tests
set "SCRIPT="
set "ARGS="
:next_test_argument
if "%~1"=="" goto invoke_tests
set "ARGS=%ARGS% %1"
shift
goto next_test_argument

:invoke_tests
"%PY%" -m pytest tests %ARGS%
exit /b %errorlevel%

:usage
echo usage: willspeak [setup^|run^|check^|fetch-models^|bench^|evaluate^|train^|export^|torgo^|match-eval^|test^|train-setup] [args...]
echo   setup [--no-models]  first-time install: .venv, requirements and tests, Parakeet
exit /b 0

:setup
if exist ".venv\Scripts\python.exe" goto setup_install
echo == creating .venv
py -3.12 -m venv .venv >nul 2>nul
if not exist ".venv\Scripts\python.exe" py -3.13 -m venv .venv >nul 2>nul
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv >nul 2>nul
if exist ".venv\Scripts\python.exe" goto setup_install
python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul
if errorlevel 1 goto setup_no_python
python -m venv .venv
if not exist ".venv\Scripts\python.exe" goto setup_no_python

:setup_install
set "PY=%CD%\.venv\Scripts\python.exe"
echo == installing requirements (runtime + tests) into .venv
"%PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%PY%" -m pip install -r requirements-dev.txt
if errorlevel 1 exit /b 1
if /i "%~2"=="--no-models" goto setup_done
echo == downloading Parakeet (once; a few hundred MB)
"%PY%" scripts\fetch_models.py --parakeet-only
if errorlevel 1 exit /b 1

:setup_done
echo.
echo app environment ready (.venv). Next:
echo   willspeak test          run the tests
echo   willspeak run           start the app, then open http://127.0.0.1:8000/live
echo   Training runs in WSL2 (Ubuntu), not Windows: see ..\docs\training.md section 2.
exit /b 0

:setup_no_python
echo Python 3.11 or newer was not found. Install Python 3.12 from python.org
echo (tick "Add python.exe to PATH"), then run "willspeak setup" again.
exit /b 1

:train_setup
echo NeMo training does not run on native Windows. Open Ubuntu (WSL2), then:
echo   git clone ^<this repository^> ~/willspeak
echo   cd ~/willspeak/app
echo   ./willspeak.sh setup
echo   ./willspeak.sh train-setup
echo Details: ..\docs\training.md section 2.
exit /b 1

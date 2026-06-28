@echo off
chcp 65001 > nul
cd /d "%~dp0"

py -3 --version >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  set "PY=python"
)

echo.
echo Установка зависимостей из requirements.txt...
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Не удалось установить зависимости. Проверьте, что установлен Python 3.10+ и pip.
  pause
  exit /b 1
)

echo.
echo Запуск веб-интерфейса: http://127.0.0.1:8765/
echo Чтобы остановить программу, закройте это окно или нажмите Ctrl+C.
echo.
%PY% web_app.py --host 127.0.0.1 --port 8765
pause

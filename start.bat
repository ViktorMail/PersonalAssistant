@echo off
title AI Grid Trainer - Setup
color 0A

echo ========================================
echo    AI GRID TRAINER - УСТАНОВКА
echo ========================================
echo.

cd /d "C:\Projects\PersonalAssistant"

if not exist "C:\Projects\PersonalAssistant" (
    echo [ERROR] Путь не найден!
    pause
    exit /b 1
)

:: ====== ИЩЕМ PYTHON ======
echo [1/4] Поиск Python...

set PYTHON_PATH=

:: Пробуем стандартные пути
if exist "C:\Users\pilig_s3awhah\AppData\Local\Programs\Python\Python310\python.exe" (
    set PYTHON_PATH=C:\Users\pilig_s3awhah\AppData\Local\Programs\Python\Python310\python.exe
)

if "%PYTHON_PATH%"=="" (
    if exist "C:\Users\pilig_s3awhah\AppData\Local\Programs\Python\Python39\python.exe" (
        set PYTHON_PATH=C:\Users\pilig_s3awhah\AppData\Local\Programs\Python\Python39\python.exe
    )
)

if "%PYTHON_PATH%"=="" (
    if exist "C:\Python310\python.exe" (
        set PYTHON_PATH=C:\Python310\python.exe
    )
)

if "%PYTHON_PATH%"=="" (
    if exist "C:\Python39\python.exe" (
        set PYTHON_PATH=C:\Python39\python.exe
    )
)

if "%PYTHON_PATH%"=="" (
    if exist "C:\Users\pilig_s3awhah\AppData\Local\Microsoft\WindowsApps\python.exe" (
        set PYTHON_PATH=C:\Users\pilig_s3awhah\AppData\Local\Microsoft\WindowsApps\python.exe
    )
)

if "%PYTHON_PATH%"=="" (
    echo [ERROR] Python не найден!
    echo.
    echo Установи Python с https://www.python.org/downloads/
    echo ИЛИ укажи путь вручную
    pause
    exit /b 1
)

echo    ✅ Python найден: %PYTHON_PATH%
%PYTHON_PATH% --version

:: ====== ДОБАВЛЯЕМ В PATH ======
echo [2/4] Добавление Python в PATH...

:: Получаем папку Python (без python.exe)
for %%i in ("%PYTHON_PATH%") do set PYTHON_DIR=%%~dpi

:: Добавляем в PATH
setx PATH "%PYTHON_DIR%;%PYTHON_DIR%Scripts;%PATH%" >nul

echo    ✅ Python добавлен в PATH!

:: ====== УСТАНАВЛИВАЕМ БИБЛИОТЕКИ ======
echo [3/4] Установка библиотек...
%PYTHON_PATH% -m pip install pandas numpy tensorflow scikit-learn matplotlib joblib -q

:: ====== ПРОВЕРЯЕМ ФАЙЛЫ ======
echo [4/4] Проверка файлов...

if exist "run.py" ( echo    [OK] run.py ) else ( echo    [MISS] run.py )
if exist "grid_ai_trainer5.py" ( echo    [OK] grid_ai_trainer5.py ) else ( echo    [MISS] grid_ai_trainer5.py )
if exist "AI_Data" ( echo    [OK] AI_Data ) else ( echo    [WARN] AI_Data не найдена )

echo.
echo ========================================
echo    ✅ УСТАНОВКА ЗАВЕРШЕНА!
echo ========================================
echo.
echo ⚠️ ПЕРЕЗАПУСТИТЕ ТЕРМИНАЛ VSCode!
echo    Закройте и откройте заново.
echo.
echo ========================================
echo    КОМАНДЫ ПОСЛЕ ПЕРЕЗАПУСКА:
echo ========================================
echo.
echo   cd C:\Projects\PersonalAssistant
echo   python run.py
echo.
echo ========================================
echo.

pause
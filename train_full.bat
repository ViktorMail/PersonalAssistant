@echo off
chcp 65001 > nul
title Grid AI Trainer PRO
color 0B

:: ========================================
:: Grid AI Trainer PRO - Полное управление
:: Author: @ViktorKon
:: ========================================

:MENU
cls
echo ========================================
echo       GRID AI TRAINER PRO v1.0
echo       Author: @ViktorKon
echo ========================================
echo.
echo [1] 🚀 Запустить обучение
echo [2] 📊 Показать статистику данных
echo [3] 📁 Выбрать другой файл данных
echo [4] 🔧 Пересоздать окружение
echo [5] 🧪 Тестовый запуск (быстрая проверка)
echo [6] 📦 Экспортировать веса для MT4
echo [7] ❌ Выход
echo.
set /p choice="Выберите действие (1-7): "

if "%choice%"=="1" goto TRAIN
if "%choice%"=="2" goto STATS
if "%choice%"=="3" goto SELECT_FILE
if "%choice%"=="4" goto RECREATE
if "%choice%"=="5" goto TEST
if "%choice%"=="6" goto EXPORT
if "%choice%"=="7" goto EXIT

echo Неверный выбор!
timeout /t 2 >nul
goto MENU

:RECREATE
cls
echo [🔄] Пересоздание окружения...
if exist "tf_env\" (
    echo Удаление старого окружения...
    rmdir /s /q tf_env
)
echo Создание нового окружения...
python -m venv tf_env
call tf_env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install tensorflow pandas numpy scikit-learn matplotlib joblib
echo [✅] Готово!
timeout /t 3 >nul
goto MENU

:SELECT_FILE
cls
echo Доступные CSV файлы:
echo.
dir *.csv /b 2>nul
if %errorlevel% neq 0 (
    echo [❌] CSV файлы не найдены!
    timeout /t 3 >nul
    goto MENU
)
echo.
set /p DATA_FILE="Введите имя файла для обучения: "
echo %DATA_FILE% > current_data.txt
echo [✅] Выбран файл: %DATA_FILE%
timeout /t 2 >nul
goto MENU

:TRAIN
cls
echo ========================================
echo        ЗАПУСК ОБУЧЕНИЯ
echo ========================================
echo.

:: Проверяем наличие файла данных
if exist current_data.txt (
    set /p DATA_FILE=<current_data.txt
) else (
    set DATA_FILE=grid_training_data.csv
)

if not exist "%DATA_FILE%" (
    echo [❌] Файл %DATA_FILE% не найден!
    echo.
    dir *.csv /b 2>nul
    echo.
    pause
    goto MENU
)

:: Создаем окружение если нет
if not exist "tf_env\" (
    echo [1/5] Создание виртуального окружения...
    python -m venv tf_env
)

:: Активируем и устанавливаем зависимости
echo [2/5] Активация окружения...
call tf_env\Scripts\activate.bat

echo [3/5] Проверка зависимостей...
python -c "import tensorflow" 2>nul
if %errorlevel% neq 0 (
    echo Установка TensorFlow и зависимостей...
    pip install tensorflow pandas numpy scikit-learn matplotlib joblib
)

:: Параметры обучения
echo.
echo Параметры обучения:
echo   Файл данных: %DATA_FILE%
echo   Эпохи: 100 (по умолчанию)
echo   Batch size: 32
echo   Learning rate: 0.001
echo.

:: Создаем временный скрипт с настройками
echo [4/5] Запуск обучения...
python -c "
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import matplotlib.pyplot as plt

print('Загрузка данных...')
data = pd.read_csv('%DATA_FILE%')
print(f'Загружено {len(data)} записей')

feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 
                'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 
                'bb_lower', 'atr', 'grid_level', 'distance_to_grid',
                'avg_grid_distance', 'orders_count', 'hour', 'minute', 'day']

X = data[feature_cols].values
y = data['target'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {len(X_train)}, Val: {len(X_val)}')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
joblib.dump(scaler, 'grid_scaler.pkl')

model = models.Sequential([
    tf.keras.Input(shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='tanh'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
                    epochs=100, batch_size=32, verbose=1)

model.save('best_model.keras')

weights = []
for layer in model.layers:
    if hasattr(layer, 'get_weights'):
        w = layer.get_weights()
        for arr in w:
            weights.extend(arr.flatten())

with open('grid_model_weights.csv', 'w') as f:
    for w in weights:
        f.write(f'{w:.10f}\n')

print(f'Экспортировано {len(weights)} весов')
"

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo [✅] ОБУЧЕНИЕ ЗАВЕРШЕНО!
    echo ========================================
    echo.
    echo Файлы созданы:
    dir grid_model_weights.csv /b 2>nul && echo   - grid_model_weights.csv
    dir best_model.keras /b 2>nul && echo   - best_model.keras
    dir training_history.png /b 2>nul && echo   - training_history.png
    echo.
) else (
    echo.
    echo [❌] Ошибка обучения!
)

deactivate
echo.
pause
goto MENU

:STATS
cls
echo ========================================
echo        СТАТИСТИКА ДАННЫХ
echo ========================================
echo.

if exist current_data.txt (
    set /p DATA_FILE=<current_data.txt
) else (
    set DATA_FILE=grid_training_data.csv
)

if not exist "%DATA_FILE%" (
    echo [❌] Файл %DATA_FILE% не найден!
    pause
    goto MENU
)

python -c "
import pandas as pd
data = pd.read_csv('%DATA_FILE%')
print(f'\nФайл: %DATA_FILE%')
print(f'Всего записей: {len(data)}')
print(f'Колонки: {list(data.columns)}')
if 'target' in data.columns:
    print('\nРаспределение target:')
    for val, count in data['target'].value_counts().items():
        name = {0:'BUY',1:'SELL',2:'HOLD'}.get(val, f'Class {val}')
        print(f'  {name}: {count} ({count/len(data)*100:.1f}%)')
print('\nПервые 5 строк:')
print(data.head())
input('\nНажмите Enter для продолжения...')
"

pause
goto MENU

:TEST
cls
echo ========================================
echo        ТЕСТОВЫЙ ЗАПУСК
echo ========================================
echo.

if not exist "tf_env\" (
    python -m venv tf_env
)

call tf_env\Scripts\activate.bat
pip install tensorflow >nul 2>nul

python -c "
import tensorflow as tf
print(f'TensorFlow версия: {tf.__version__}')
print('✅ TensorFlow работает!')
input('Нажмите Enter для продолжения...')
"

deactivate
goto MENU

:EXPORT
cls
echo ========================================
echo        ЭКСПОРТ ВЕСОВ ДЛЯ MT4
echo ========================================
echo.

if exist "grid_model_weights.csv" (
    echo [✅] Файл весов найден: grid_model_weights.csv
    echo.
    echo Куда скопировать файл?
    echo [1] В папку MQL4\Files текущего терминала
    echo [2] Выбрать папку вручную
    echo [3] Отмена
    echo.
    set /p exp_choice="Выберите (1-3): "
    
    if "!exp_choice!"=="1" (
        for /d %%i in ("%AppData%\MetaQuotes\Terminal\*") do (
            if exist "%%i\MQL4\Files" (
                copy grid_model_weights.csv "%%i\MQL4\Files\" >nul
                echo [✅] Скопировано в %%i\MQL4\Files\
            )
        )
    )
    if "!exp_choice!"=="2" (
        set /p mql_path="Введите путь к папке MQL4\Files: "
        copy grid_model_weights.csv "!mql_path!\" >nul
        echo [✅] Скопировано
    )
) else (
    echo [❌] Файл grid_model_weights.csv не найден!
    echo Сначала запустите обучение.
)

pause
goto MENU

:EXIT
echo.
echo До свидания!
timeout /t 2 >nul
exit /b 0
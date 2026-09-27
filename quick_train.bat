@echo off
chcp 65001 > nul
echo Запуск обучения Grid AI Trainer...

:: Проверяем файл данных
if not exist "grid_training_data.csv" (
    echo Файл grid_training_data.csv не найден!
    dir *.csv
    set /p file="Введите имя CSV файла: "
    set DATA_FILE=%file%
) else (
    set DATA_FILE=grid_training_data.csv
)

:: Создаем/активируем окружение
if not exist "tf_env\" python -m venv tf_env
call tf_env\Scripts\activate.bat

:: Устанавливаем зависимости
pip install tensorflow pandas numpy scikit-learn matplotlib joblib

:: Запускаем обучение
python grid_ai_trainer.py --data %DATA_FILE%

:: Копируем веса в MT4 если найдем
if exist "grid_model_weights.csv" (
    for /d %%i in ("%AppData%\MetaQuotes\Terminal\*") do (
        if exist "%%i\MQL4\Files" (
            copy grid_model_weights.csv "%%i\MQL4\Files\" >nul
            echo Веса скопированы в MT4
        )
    )
)

pause
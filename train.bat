@echo off
chcp 65001 > nul
echo ========================================
echo   GRID AI TRAINER - ЗАПУСК ОБУЧЕНИЯ
echo   Author: @ViktorKon
echo ========================================
echo.

:: Активируем окружение
call tf_env\Scripts\activate.bat

:: Запускаем обучение
python grid_ai_trainer.py --market forex --data grid_training_data.csv

pause
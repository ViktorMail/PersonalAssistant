@echo off
echo Building Grid AI Trainer EXE...
echo.

pyinstaller --onefile --windowed ^
    --name "GridAITrainer" ^
    --icon icon.ico ^
    --add-data "icon.ico;." ^
    --hidden-import pandas ^
    --hidden-import numpy ^
    --hidden-import tensorflow ^
    --hidden-import sklearn ^
    --hidden-import joblib ^
    --hidden-import matplotlib ^
    grid_ai_trainer_gui.py

echo.
echo Build complete! Check the 'dist' folder.
pause
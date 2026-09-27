#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Strategy Trainer - Универсальное обучение
Автор: @ViktorKon
Автоматическое определение таймфрейма из имени файла
"""

import os
import sys
import argparse
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Отключаем предупреждения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# Импорты для TensorFlow
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Параметры по умолчанию
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT = 0.3
DEFAULT_L2 = 0.0001

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_color(text, color=Colors.END, bold=False):
    if bold:
        print(f"{Colors.BOLD}{color}{text}{Colors.END}")
    else:
        print(f"{color}{text}{Colors.END}")

def extract_info_from_filename(filename):
    """
    Извлекает символ, таймфрейм и тип рынка из имени файла
    Поддерживает форматы:
    - training_EURUSD_M5.csv
    - training_GBPUSD_H1_forex.csv
    - training_BTCUSD_M5_crypto.csv
    - EURUSD_M5_training.csv
    """
    filename = Path(filename).stem  # Убираем расширение
    
    # Паттерны для поиска
    patterns = [
        # training_EURUSD_M5.csv
        r'training_([A-Z]+)_([A-Z0-9]+)',
        # training_EURUSD_M5_forex.csv
        r'training_([A-Z]+)_([A-Z0-9]+)_([a-z]+)',
        # EURUSD_M5_training.csv
        r'([A-Z]+)_([A-Z0-9]+)_training',
        # EURUSD_M5.csv
        r'([A-Z]+)_([A-Z0-9]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                symbol = groups[0]
                timeframe = groups[1]
                market = None
            elif len(groups) == 3:
                symbol = groups[0]
                timeframe = groups[1]
                market = groups[2]
            else:
                continue
            
            # Проверяем, что timeframe выглядит как таймфрейм
            if timeframe in ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']:
                # Если market не определен, пытаемся угадать
                if market is None:
                    if symbol in ['BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD']:
                        market = 'crypto'
                    elif symbol in ['XAUUSD', 'XAGUSD', 'CL']:
                        market = 'commodities'
                    else:
                        market = 'forex'
                
                return symbol, timeframe, market
    
    return None, None, None

def get_config(market_type, timeframe):
    """Получить конфигурацию для рынка и таймфрейма"""
    
    # Базовые конфигурации по рынкам
    base_configs = {
        'forex': {
            'name': 'Forex',
            'threshold': 0.0002,
            'dropout': 0.3,
            'hidden_layers': [64, 128, 64, 32],
            'learning_rate': 0.001
        },
        'crypto': {
            'name': 'Crypto',
            'threshold': 0.005,
            'dropout': 0.4,
            'hidden_layers': [128, 256, 128, 64],
            'learning_rate': 0.002
        },
        'commodities': {
            'name': 'Commodities',
            'threshold': 0.001,
            'dropout': 0.3,
            'hidden_layers': [64, 128, 64, 32],
            'learning_rate': 0.001
        }
    }
    
    config = base_configs.get(market_type, base_configs['forex'])
    
    # Корректируем параметры в зависимости от таймфрейма
    tf_adjustments = {
        'M1': {'threshold': 0.0001, 'dropout': 0.35, 'epochs': 150},
        'M5': {'threshold': 0.0002, 'dropout': 0.3, 'epochs': 100},
        'M15': {'threshold': 0.0003, 'dropout': 0.3, 'epochs': 100},
        'M30': {'threshold': 0.0004, 'dropout': 0.25, 'epochs': 100},
        'H1': {'threshold': 0.0005, 'dropout': 0.25, 'epochs': 100},
        'H4': {'threshold': 0.001, 'dropout': 0.2, 'epochs': 100},
        'D1': {'threshold': 0.002, 'dropout': 0.2, 'epochs': 100},
    }
    
    if timeframe in tf_adjustments:
        adj = tf_adjustments[timeframe]
        config['threshold'] = adj['threshold']
        config['dropout'] = adj['dropout']
    
    return config

def load_data(filename, market_type, symbol):
    """Загрузка данных"""
    print_color(f"\n📊 Загрузка данных для {market_type.upper()} | {symbol}", Colors.CYAN)
    print(f"   Файл: {filename}")
    
    try:
        data = pd.read_csv(filename)
        print(f"   Записей: {len(data)}")
        print(f"   Колонки: {list(data.columns)}")
        
        # Определяем доступные колонки
        feature_cols = []
        
        # Базовые ценовые колонки
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                feature_cols.append(col)
        
        # Индикаторы
        for col in ['rsi', 'macd', 'macd_signal', 'macd_hist', 
                    'bb_upper', 'bb_middle', 'bb_lower', 'atr']:
            if col in data.columns:
                feature_cols.append(col)
        
        # Временные колонки
        for col in ['hour', 'minute', 'day_of_week']:
            if col in data.columns:
                feature_cols.append(col)
        
        print(f"   Используется {len(feature_cols)} признаков")
        
        X = data[feature_cols].values
        
        # Создаем target
        if 'target' in data.columns:
            y = data['target'].values
        else:
            print("   🎯 Создание target на основе future price...")
            future_returns = data['close'].pct_change(5).shift(-5)
            y = np.zeros(len(data))
            threshold = get_config(market_type, 'H1')['threshold']
            y[future_returns > threshold] = 0
            y[future_returns < -threshold] = 1
            y[np.abs(future_returns) <= threshold] = 2
        
        # Удаляем NaN
        valid_idx = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx].astype(int)
        
        return X, y, feature_cols
        
    except Exception as e:
        print_color(f"❌ Ошибка загрузки: {e}", Colors.RED)
        return None, None, None

def create_model(input_size, market_type, timeframe):
    """Создание модели с адаптацией под рынок и таймфрейм"""
    
    config = get_config(market_type, timeframe)
    hidden = config['hidden_layers']
    dropout_rate = config['dropout']
    
    model = models.Sequential([
        tf.keras.Input(shape=(input_size,)),
        
        layers.Dense(hidden[0], kernel_regularizer=regularizers.l2(DEFAULT_L2)),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout_rate),
        
        layers.Dense(hidden[1], kernel_regularizer=regularizers.l2(DEFAULT_L2)),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout_rate),
        
        layers.Dense(hidden[2], kernel_regularizer=regularizers.l2(DEFAULT_L2)),
        layers.BatchNormalization(),
        layers.ELU(alpha=1.0),
        layers.Dropout(dropout_rate * 0.7),
        
        layers.Dense(hidden[3], kernel_regularizer=regularizers.l2(DEFAULT_L2)),
        layers.BatchNormalization(),
        layers.Activation('tanh'),
        
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def export_weights_to_mql4(model, filename):
    """Экспорт весов для MT4"""
    weights_list = []
    
    for layer in model.layers:
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            for w in weights:
                weights_list.extend(w.flatten())
    
    with open(filename, 'w') as f:
        for w in weights_list:
            f.write(f"{w:.10f}\n")
    
    print_color(f"✅ Экспортировано {len(weights_list)} весов в {filename}", Colors.GREEN)

def plot_training_history(history, symbol, timeframe, market, filename):
    """Построение графика обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title(f'{symbol} {timeframe} - Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_title(f'{symbol} {timeframe} - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print_color(f"✅ График сохранен в {filename}", Colors.GREEN)

def main():
    parser = argparse.ArgumentParser(description='AI Strategy Trainer')
    parser.add_argument('--data', type=str, help='CSV файл с данными')
    parser.add_argument('--market', type=str, help='Тип рынка (forex/crypto/commodities)')
    parser.add_argument('--symbol', type=str, help='Торговый символ')
    parser.add_argument('--timeframe', type=str, help='Таймфрейм (M5, H1, H4)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Количество эпох')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE, help='Размер батча')
    
    args = parser.parse_args()
    
    print_color("=" * 60, Colors.CYAN, True)
    print_color("   AI STRATEGY TRAINER - УНИВЕРСАЛЬНОЕ ОБУЧЕНИЕ", Colors.CYAN, True)
    print_color("=" * 60, Colors.CYAN, True)
    
    # Определяем файл данных
    if args.data:
        data_file = args.data
    else:
        csv_files = list(Path(".").glob("*.csv"))
        if not csv_files:
            print_color("❌ Нет CSV файлов!", Colors.RED)
            return
        print_color("\n📁 Доступные файлы:", Colors.CYAN)
        for i, f in enumerate(csv_files, 1):
            size = f.stat().st_size / 1024
            print(f"   [{i}] {f.name} ({size:.1f} KB)")
        choice = int(input("\nВыберите файл: ")) - 1
        data_file = str(csv_files[choice])
    
    # Извлекаем информацию из имени файла
    symbol, timeframe, market = extract_info_from_filename(data_file)
    
    # Используем переданные параметры, если они есть
    if args.symbol:
        symbol = args.symbol
    if args.timeframe:
        timeframe = args.timeframe
    if args.market:
        market = args.market
    
    if not symbol or not timeframe or not market:
        print_color("❌ Не удалось определить параметры из имени файла!", Colors.RED)
        print(f"   Файл: {data_file}")
        print("   Имя файла должно содержать символ и таймфрейм, например: training_EURUSD_M5.csv")
        return
    
    print_color(f"\n📊 ПАРАМЕТРЫ ОБУЧЕНИЯ:", Colors.CYAN, True)
    print(f"   Рынок: {market.upper()}")
    print(f"   Символ: {symbol}")
    print(f"   Таймфрейм: {timeframe}")
    print(f"   Файл: {data_file}")
    print(f"   Эпохи: {args.epochs}")
    print(f"   Батч: {args.batch}")
    
    # Загрузка данных
    X, y, features = load_data(data_file, market, symbol)
    if X is None:
        return
    
    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Сохраняем scaler
    scaler_file = f"scaler_{symbol}_{timeframe}_{market}.pkl"
    joblib.dump(scaler, scaler_file)
    print_color(f"✅ Scaler сохранен в {scaler_file}", Colors.GREEN)
    
    # Создание модели
    print_color("\n🧠 Создание модели...", Colors.CYAN)
    input_size = X.shape[1]
    model = create_model(input_size, market, timeframe)
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(f'best_model_{symbol}_{timeframe}_{market}.keras', 
                    monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Обучение
    print_color("\n🚀 Запуск обучения...", Colors.GREEN, True)
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=1
    )
    
    # Оценка
    train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)[1]
    val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)[1]
    
    print_color("\n" + "=" * 60, Colors.CYAN, True)
    print_color("📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ", Colors.CYAN, True)
    print_color("=" * 60, Colors.CYAN, True)
    print_color(f"   Train Accuracy: {train_acc:.4f}", Colors.GREEN)
    print_color(f"   Val Accuracy:   {val_acc:.4f}", Colors.GREEN)
    print_color(f"   Разрыв:         {abs(train_acc - val_acc):.4f}", 
                Colors.YELLOW if abs(train_acc - val_acc) < 0.1 else Colors.RED)
    
    # Сохранение модели и весов
    model_file = f"model_{symbol}_{timeframe}_{market}.keras"
    weights_file = f"model_{symbol}_{timeframe}_{market}_weights.csv"
    plot_file = f"training_{symbol}_{timeframe}_{market}.png"
    
    model.save(model_file)
    print_color(f"✅ Модель сохранена: {weights_file}", Colors.GREEN)
    
    export_weights_to_mql4(model, weights_file)
    plot_training_history(history, symbol, timeframe, market, plot_file)
    
    print_color("\n" + "=" * 60, Colors.CYAN, True)
    print_color("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!", Colors.GREEN, True)
    print_color("=" * 60, Colors.CYAN, True)
    print_color(f"\n📌 Для MT4 используйте файл: {weights_file}", Colors.YELLOW)
    print_color(f"   Скопируйте его в папку MQL4\\Files\\", Colors.CYAN)

if __name__ == "__main__":
    main()
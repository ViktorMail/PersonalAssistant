#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Strategy Trainer V4 - Расширенное обучение для скальпинга
Добавлены признаки: трендовые линии, объёмы, ATR, RSI
Автор: @ViktorKon
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ===== НОВЫЕ УВЕЛИЧЕННЫЕ ПАРАМЕТРЫ =====
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_DROPOUT = 0.3
DEFAULT_L2 = 0.0001

# Увеличенные размеры слоёв для скальпинга
DEFAULT_HIDDEN1 = 256
DEFAULT_HIDDEN2 = 512
DEFAULT_HIDDEN3 = 256
DEFAULT_HIDDEN4 = 128

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
    """Извлекает символ, таймфрейм и тип рынка из имени файла"""
    filename = Path(filename).stem
    patterns = [
        r'training_([A-Z]+)_([A-Z0-9]+)',
        r'training_([A-Z]+)_([A-Z0-9]+)_([a-z]+)',
        r'([A-Z]+)_([A-Z0-9]+)_training',
        r'([A-Z]+)_([A-Z0-9]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                symbol, timeframe = groups[0], groups[1]
                market = None
            elif len(groups) == 3:
                symbol, timeframe, market = groups
            else:
                continue
            
            if timeframe in ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']:
                if market is None:
                    if symbol in ['BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD']:
                        market = 'crypto'
                    elif symbol in ['XAUUSD', 'XAGUSD', 'CL']:
                        market = 'commodities'
                    else:
                        market = 'forex'
                return symbol, timeframe, market
    
    return None, None, None

def calculate_trendline_features(data):
    """
    Расчёт признаков на основе трендовых линий
    Возвращает: расстояние до линии, направление, угол наклона
    """
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    
    n = len(data)
    trend_distances = np.zeros(n)
    trend_directions = np.zeros(n)
    trend_angles = np.zeros(n)
    
    for i in range(20, n):
        # Поиск локальных экстремумов на 20 барах
        window_highs = highs[i-20:i]
        window_lows = lows[i-20:i]
        
        # Медвежья трендовая линия (по хаям)
        high_indices = []
        high_values = []
        for j in range(len(window_highs)-2):
            if (window_highs[j] > window_highs[j-1] and 
                window_highs[j] > window_highs[j+1]):
                high_indices.append(j)
                high_values.append(window_highs[j])
        
        # Бычья трендовая линия (по лоям)
        low_indices = []
        low_values = []
        for j in range(len(window_lows)-2):
            if (window_lows[j] < window_lows[j-1] and 
                window_lows[j] < window_lows[j+1]):
                low_indices.append(j)
                low_values.append(window_lows[j])
        
        # Расчёт расстояния до ближайшей трендовой линии
        if len(high_indices) >= 2:
            x1, x2 = high_indices[-2], high_indices[-1]
            y1, y2 = high_values[-2], high_values[-1]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            line_value = y1 + slope * (len(window_highs) - x1)
            trend_distances[i] = (closes[i] - line_value) / (data['atr'].iloc[i] + 0.0001)
            trend_directions[i] = -1 if slope < 0 else 1
            trend_angles[i] = np.arctan(slope) if abs(slope) < 10 else np.pi/2
        
        if len(low_indices) >= 2 and abs(trend_distances[i]) < 0.1:
            x1, x2 = low_indices[-2], low_indices[-1]
            y1, y2 = low_values[-2], low_values[-1]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            line_value = y1 + slope * (len(window_lows) - x1)
            dist = (closes[i] - line_value) / (data['atr'].iloc[i] + 0.0001)
            if abs(dist) < abs(trend_distances[i]):
                trend_distances[i] = dist
                trend_directions[i] = 1 if slope > 0 else -1
                trend_angles[i] = np.arctan(slope) if abs(slope) < 10 else np.pi/2
    
    return trend_distances, trend_directions, trend_angles

def load_data(filename, market_type, symbol, timeframe):
    """Загрузка данных с расширенными признаками"""
    print_color(f"\n📊 Загрузка данных для {market_type.upper()} | {symbol} | {timeframe}", Colors.CYAN)
    print(f"   Файл: {filename}")
    
    try:
        data = pd.read_csv(filename)
        print(f"   Записей: {len(data)}")
        
        # Базовые признаки
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                        'rsi', 'macd', 'macd_signal', 'macd_hist',
                        'bb_upper', 'bb_middle', 'bb_lower', 'atr',
                        'hour', 'minute', 'day_of_week']
        
        # Добавляем трендовые признаки
        trend_dist, trend_dir, trend_angle = calculate_trendline_features(data)
        data['trend_distance'] = trend_dist
        data['trend_direction'] = trend_dir
        data['trend_angle'] = trend_angle
        
        feature_cols.extend(['trend_distance', 'trend_direction', 'trend_angle'])
        
        print(f"   Используется {len(feature_cols)} признаков")
        
        X = data[feature_cols].values
        
        # Настройки для скальпинга (M5)
        lookahead = 12  # 60 минут
        threshold = 0.05  # 0.05% для M5
        
        if timeframe == 'M15':
            lookahead = 8
            threshold = 0.08
        elif timeframe == 'H1':
            lookahead = 6
            threshold = 0.15
        
        # Создаём target
        if 'target' in data.columns:
            y = data['target'].values
        else:
            print(f"   🎯 Создание target (lookahead={lookahead}, threshold={threshold}%)")
            future_returns = data['close'].pct_change(lookahead).shift(-lookahead)
            y = np.zeros(len(data))
            y[future_returns > threshold/100] = 0
            y[future_returns < -threshold/100] = 1
            y[np.abs(future_returns) <= threshold/100] = 2
        
        valid_idx = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx].astype(int)
        
        print(f"   BUY: {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
        print(f"   SELL: {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
        print(f"   HOLD: {np.sum(y==2)} ({np.sum(y==2)/len(y)*100:.1f}%)")
        
        return X, y, feature_cols
        
    except Exception as e:
        print_color(f"❌ Ошибка загрузки: {e}", Colors.RED)
        return None, None, None

def create_scalping_model(input_size, market_type, timeframe):
    """Создание модели специально для скальпинга с увеличенной множественностью"""
    
    # Адаптивные настройки для скальпинга
    if market_type == 'crypto':
        hidden = [384, 768, 384, 192]
        dropout = 0.4
        lr = 0.0003
    elif market_type == 'commodities':
        hidden = [320, 640, 320, 160]
        dropout = 0.35
        lr = 0.0004
    else:
        hidden = [DEFAULT_HIDDEN1, DEFAULT_HIDDEN2, DEFAULT_HIDDEN3, DEFAULT_HIDDEN4]
        dropout = DEFAULT_DROPOUT
        lr = DEFAULT_LEARNING_RATE
    
    model = models.Sequential([
        tf.keras.Input(shape=(input_size,)),
        
        layers.Dense(hidden[0], kernel_regularizer=regularizers.l2(DEFAULT_L2)),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout),
        
        layers.Dense(hidden[1], kernel_regularizer=regularizers.l2(DEFAULT_L2)),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout),
        
        layers.Dense(hidden[2], kernel_regularizer=regularizers.l2(DEFAULT_L2)),
        layers.BatchNormalization(),
        layers.ELU(alpha=1.0),
        layers.Dropout(dropout * 0.7),
        
        layers.Dense(hidden[3], kernel_regularizer=regularizers.l2(DEFAULT_L2)),
        layers.BatchNormalization(),
        layers.Activation('tanh'),
        
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title(f'{symbol} {timeframe} - Model Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_title(f'{symbol} {timeframe} - Model Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print_color(f"✅ График сохранен в {filename}", Colors.GREEN)

def main():
    parser = argparse.ArgumentParser(description='AI Strategy Trainer V4 - Скальпинг')
    parser.add_argument('--data', type=str, help='CSV файл с данными')
    parser.add_argument('--market', type=str, help='Тип рынка (forex/crypto/commodities)')
    parser.add_argument('--symbol', type=str, help='Торговый символ')
    parser.add_argument('--timeframe', type=str, help='Таймфрейм (M5, M15, H1)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Количество эпох')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE, help='Размер батча')
    
    args = parser.parse_args()
    
    print_color("=" * 70, Colors.CYAN, True)
    print_color("   AI STRATEGY TRAINER V4 - СКАЛЬПИНГ", Colors.CYAN, True)
    print_color("=" * 70, Colors.CYAN, True)
    
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
    
    symbol, timeframe, market = extract_info_from_filename(data_file)
    
    if args.symbol:
        symbol = args.symbol
    if args.timeframe:
        timeframe = args.timeframe
    if args.market:
        market = args.market
    
    if not symbol or not timeframe or not market:
        print_color("❌ Не удалось определить параметры из имени файла!", Colors.RED)
        return
    
    print_color(f"\n📊 ПАРАМЕТРЫ ОБУЧЕНИЯ:", Colors.CYAN, True)
    print(f"   Рынок: {market.upper()}")
    print(f"   Символ: {symbol}")
    print(f"   Таймфрейм: {timeframe}")
    print(f"   Файл: {data_file}")
    print(f"   Эпохи: {args.epochs}")
    print(f"   Батч: {args.batch}")
    
    X, y, features = load_data(data_file, market, symbol, timeframe)
    if X is None:
        return
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    scaler_file = f"scaler_{symbol}_{timeframe}_{market}.pkl"
    joblib.dump(scaler, scaler_file)
    print_color(f"✅ Scaler сохранен в {scaler_file}", Colors.GREEN)
    
    print_color("\n🧠 Создание модели для скальпинга...", Colors.CYAN)
    model = create_scalping_model(X.shape[1], market, timeframe)
    model.summary()
    
    total_params = model.count_params()
    print_color(f"\n📊 Всего параметров: {total_params:,}", Colors.YELLOW)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f'best_model_{symbol}_{timeframe}_{market}.keras', 
                        monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    print_color("\n🚀 Запуск обучения...", Colors.GREEN, True)
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=1
    )
    
    train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)[1]
    val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)[1]
    
    print_color("\n" + "=" * 70, Colors.CYAN, True)
    print_color("📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ", Colors.CYAN, True)
    print_color("=" * 70, Colors.CYAN, True)
    print_color(f"   Train Accuracy: {train_acc:.4f}", Colors.GREEN)
    print_color(f"   Val Accuracy:   {val_acc:.4f}", Colors.GREEN)
    print_color(f"   Разрыв:         {abs(train_acc - val_acc):.4f}", 
                Colors.YELLOW if abs(train_acc - val_acc) < 0.1 else Colors.RED)
    
    model_file = f"model_{symbol}_{timeframe}_{market}.keras"
    weights_file = f"model_{symbol}_{timeframe}_{market}_weights.csv"
    plot_file = f"training_{symbol}_{timeframe}_{market}.png"
    
    model.save(model_file)
    print_color(f"✅ Модель сохранена: {model_file}", Colors.GREEN)
    
    export_weights_to_mql4(model, weights_file)
    plot_training_history(history, symbol, timeframe, market, plot_file)
    
    print_color("\n" + "=" * 70, Colors.CYAN, True)
    print_color("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!", Colors.GREEN, True)
    print_color("=" * 70, Colors.CYAN, True)
    print_color(f"\n📌 Для MT4 используйте файл: {weights_file}", Colors.YELLOW)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Strategy Trainer V2 - Расширенная нейросеть
Увеличенная множественность слоёв для лучшего обучения
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

# Отключаем предупреждения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# Импорты для TensorFlow
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ===== НОВЫЕ УВЕЛИЧЕННЫЕ ПАРАМЕТРЫ =====
DEFAULT_EPOCHS = 200                    # Увеличено для лучшей сходимости
DEFAULT_BATCH_SIZE = 64                 # Увеличен для стабильности
DEFAULT_LEARNING_RATE = 0.0005          # Чуть ниже для тонкой настройки
DEFAULT_DROPOUT = 0.3
DEFAULT_L2 = 0.0001

# Увеличенные размеры слоёв
# Старая архитектура: 64-128-64-32
# Новая архитектура:   256-512-256-128
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
                symbol = groups[0]
                timeframe = groups[1]
                market = None
            elif len(groups) == 3:
                symbol = groups[0]
                timeframe = groups[1]
                market = groups[2]
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

def get_config(market_type, timeframe):
    """Получить конфигурацию для рынка и таймфрейма"""
    
    # Базовые конфигурации по рынкам
    base_configs = {
        'forex': {
            'name': 'Forex',
            'threshold': 0.0002,
            'dropout': 0.3,
            'learning_rate': 0.0005,
            'hidden1': DEFAULT_HIDDEN1,
            'hidden2': DEFAULT_HIDDEN2,
            'hidden3': DEFAULT_HIDDEN3,
            'hidden4': DEFAULT_HIDDEN4
        },
        'crypto': {
            'name': 'Crypto',
            'threshold': 0.005,
            'dropout': 0.4,
            'learning_rate': 0.0003,
            'hidden1': 384,
            'hidden2': 768,
            'hidden3': 384,
            'hidden4': 192
        },
        'commodities': {
            'name': 'Commodities',
            'threshold': 0.001,
            'dropout': 0.35,
            'learning_rate': 0.0004,
            'hidden1': 320,
            'hidden2': 640,
            'hidden3': 320,
            'hidden4': 160
        }
    }
    
    config = base_configs.get(market_type, base_configs['forex'])
    
    # Корректируем параметры в зависимости от таймфрейма
    tf_adjustments = {
        'M1': {'threshold': 0.0001, 'dropout': 0.35},
        'M5': {'threshold': 0.0002, 'dropout': 0.3},
        'M15': {'threshold': 0.0003, 'dropout': 0.3},
        'M30': {'threshold': 0.0004, 'dropout': 0.25},
        'H1': {'threshold': 0.0005, 'dropout': 0.25},
        'H4': {'threshold': 0.001, 'dropout': 0.2},
        'D1': {'threshold': 0.002, 'dropout': 0.2},
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
        
        # Определяем доступные колонки (все возможные)
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
        for col in ['hour', 'minute', 'day_of_week', 'day_of_year',
                    'hour_sin', 'hour_cos', 'volatility', 'volume_spike', 'seasonal']:
            if col in data.columns:
                feature_cols.append(col)
        
        print(f"   Используется {len(feature_cols)} признаков")
        
        X = data[feature_cols].values
        
        # Создаем target
        if 'target' in data.columns:
            y = data['target'].values
        else:
            print("   🎯 Создание target на основе future price...")
            lookahead = 12
            future_returns = data['close'].pct_change(lookahead).shift(-lookahead)
            y = np.zeros(len(data))
            threshold = get_config(market_type, 'H1')['threshold']
            y[future_returns > threshold] = 0
            y[future_returns < -threshold] = 1
            y[np.abs(future_returns) <= threshold] = 2
        
        # Удаляем NaN
        valid_idx = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx].astype(int)
        
        print(f"   Buy: {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
        print(f"   Sell: {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
        print(f"   Hold: {np.sum(y==2)} ({np.sum(y==2)/len(y)*100:.1f}%)")
        
        return X, y, feature_cols
        
    except Exception as e:
        print_color(f"❌ Ошибка загрузки: {e}", Colors.RED)
        return None, None, None

def create_wide_model(input_size, market_type, timeframe):
    """
    Создание расширенной нейросети с увеличенной множественностью слоёв
    
    Архитектура:
    - Вход: input_size (20-22 признаков)
    - Слой 1: 256-384 нейронов + BatchNorm + Dropout
    - Слой 2: 512-768 нейронов + BatchNorm + Dropout
    - Слой 3: 256-384 нейронов + BatchNorm + Dropout
    - Слой 4: 128-192 нейронов + BatchNorm
    - Выход: 3 нейрона (Buy, Sell, Hold)
    """
    
    config = get_config(market_type, timeframe)
    dropout_rate = config['dropout']
    l2_lambda = DEFAULT_L2
    
    # Для крипты усиливаем регуляризацию
    if market_type == 'crypto':
        l2_lambda = 0.0005
    
    model = models.Sequential([
        tf.keras.Input(shape=(input_size,)),
        
        # Слой 1: расширенный
        layers.Dense(config['hidden1'], 
                     kernel_regularizer=regularizers.l2(l2_lambda),
                     name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.LeakyReLU(negative_slope=0.01, name='lrelu_1'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        # Слой 2: максимально широкий
        layers.Dense(config['hidden2'],
                     kernel_regularizer=regularizers.l2(l2_lambda),
                     name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.LeakyReLU(negative_slope=0.01, name='lrelu_2'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Слой 3: средний
        layers.Dense(config['hidden3'],
                     kernel_regularizer=regularizers.l2(l2_lambda),
                     name='dense_3'),
        layers.BatchNormalization(name='bn_3'),
        layers.ELU(alpha=1.0, name='elu_3'),
        layers.Dropout(dropout_rate * 0.7, name='dropout_3'),
        
        # Слой 4: сжатие
        layers.Dense(config['hidden4'],
                     kernel_regularizer=regularizers.l2(l2_lambda),
                     name='dense_4'),
        layers.BatchNormalization(name='bn_4'),
        layers.Activation('tanh', name='tanh_4'),
        
        # Выходной слой
        layers.Dense(3, activation='softmax', name='output')
    ])
    
    # Используем AdamW для лучшей регуляризации
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config['learning_rate'],
        weight_decay=0.00001
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def export_weights_to_mql4(model, filename):
    """Экспорт весов для MT4"""
    weights_list = []
    layer_count = 0
    
    for layer in model.layers:
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            if weights:
                layer_count += 1
                for w in weights:
                    weights_list.extend(w.flatten())
    
    with open(filename, 'w') as f:
        for w in weights_list:
            f.write(f"{w:.10f}\n")
    
    print_color(f"✅ Экспортировано {len(weights_list)} весов в {filename}", Colors.GREEN)
    print(f"   Всего слоёв с весами: {layer_count}")

def plot_training_history(history, symbol, timeframe, market, filename):
    """Построение графика обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title(f'{symbol} {timeframe} - Model Loss', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_title(f'{symbol} {timeframe} - Model Accuracy', fontsize=12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print_color(f"✅ График сохранен в {filename}", Colors.GREEN)

def main():
    parser = argparse.ArgumentParser(description='AI Strategy Trainer V2 - Расширенная нейросеть')
    parser.add_argument('--data', type=str, help='CSV файл с данными')
    parser.add_argument('--market', type=str, help='Тип рынка (forex/crypto/commodities)')
    parser.add_argument('--symbol', type=str, help='Торговый символ')
    parser.add_argument('--timeframe', type=str, help='Таймфрейм (M5, H1, H4)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Количество эпох')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE, help='Размер батча')
    
    args = parser.parse_args()
    
    print_color("=" * 70, Colors.CYAN, True)
    print_color("   AI STRATEGY TRAINER V2 - РАСШИРЕННАЯ НЕЙРОСЕТЬ", Colors.CYAN, True)
    print_color("=" * 70, Colors.CYAN, True)
    
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
        print("   Имя файла должно содержать символ и таймфрейм")
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
    
    print_color(f"\n📊 Размерность данных:", Colors.CYAN)
    print(f"   Признаков: {X.shape[1]}")
    print(f"   Samples: {len(X)}")
    
    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Сохраняем scaler
    scaler_file = f"scaler_{symbol}_{timeframe}_{market}.pkl"
    joblib.dump(scaler, scaler_file)
    print_color(f"✅ Scaler сохранен в {scaler_file}", Colors.GREEN)
    
    # Создание расширенной модели
    print_color("\n🧠 Создание РАСШИРЕННОЙ модели...", Colors.CYAN)
    input_size = X.shape[1]
    model = create_wide_model(input_size, market, timeframe)
    model.summary()
    
    # Подсчёт параметров
    total_params = model.count_params()
    print_color(f"\n📊 Всего параметров: {total_params:,}", Colors.YELLOW)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f'best_model_{symbol}_{timeframe}_{market}.keras', 
                        monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Обучение
    print_color("\n🚀 Запуск обучения расширенной сети...", Colors.GREEN, True)
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
    
    print_color("\n" + "=" * 70, Colors.CYAN, True)
    print_color("📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ", Colors.CYAN, True)
    print_color("=" * 70, Colors.CYAN, True)
    print_color(f"   Train Accuracy: {train_acc:.4f}", Colors.GREEN)
    print_color(f"   Val Accuracy:   {val_acc:.4f}", Colors.GREEN)
    print_color(f"   Разрыв:         {abs(train_acc - val_acc):.4f}", 
                Colors.YELLOW if abs(train_acc - val_acc) < 0.1 else Colors.RED)
    
    # Сохранение модели и весов
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
    print_color(f"   Скопируйте его в папку MQL4\\Files\\", Colors.CYAN)
    
    # Рекомендации
    print_color("\n💡 РЕКОМЕНДАЦИИ:", Colors.CYAN)
    if val_acc > 0.72:
        print_color("   ✅ Отличная модель! Можно использовать в торговле", Colors.GREEN)
    elif val_acc > 0.68:
        print_color("   ⚠️ Хорошая модель, но требуется доработка данных", Colors.YELLOW)
    else:
        print_color("   ❌ Модель слабая, соберите больше качественных данных", Colors.RED)

if __name__ == "__main__":
    main()
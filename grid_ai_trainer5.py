#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Strategy Trainer V9 - 36 признаков + STRAT (с регуляризацией)
Поддерживает M5 и H1 данные с 36 признаками
Улучшенная регуляризация против переобучения
Автор: @ViktorKon
"""

import os
import sys
import argparse
import re
import pandas as pd
X = pd.read_csv("dataset_M5_BITCOIN_M5_X.csv")
print(f"Количество признаков: {X.shape[1]}")
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

# ===== КОНФИГУРАЦИИ ПО УМОЛЧАНИЮ =====
DEFAULT_CONFIGS = {
    'M5': {
        'lookahead': 12, 
        'threshold': 0.05, 
        'min_bars': 5000,
        'epochs': 80,
        'batch_size': 128,
        'learning_rate': 0.0005,
        'dropout': 0.3,
        'l2': 0.01,
        'hidden_layers': [32, 16, 8, 4],
        'description': 'M5 нейросеть (сильная регуляризация)'
    },
    'H1': {
        'lookahead': 5, 
        'threshold': 0.03, 
        'min_bars': 3000,
        'epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.0005,
        'dropout': 0.2,
        'l2': 0.001,
        'hidden_layers': [16,8,4, 2],
        'description': 'H1 нейросеть (сильная регуляризация)'
    }
}

class Colors:
    HEADER = '\033[95m'; BLUE = '\033[94m'; CYAN = '\033[96m'
    GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    END = '\033[0m'; BOLD = '\033[1m'

def print_color(text, color=Colors.END, bold=False):
    if bold: print(f"{Colors.BOLD}{color}{text}{Colors.END}")
    else: print(f"{color}{text}{Colors.END}")

def extract_info_from_filename(filename):
    filename = Path(filename).stem
    base_name = filename
    if filename.endswith('_X'): base_name = filename[:-2]
    elif filename.endswith('_Y'): base_name = filename[:-2]
    
    patterns = [
        r'dataset_H1_([A-Z0-9]+)_H1',
        r'dataset_M5_([A-Z0-9]+)_M5',
        r'training_([A-Z]+)_([A-Z0-9]+)_([A-Za-z\+]+)_([a-z]+)',
        r'training_([A-Z]+)_([A-Z0-9]+)_([A-Za-z\+]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, base_name)
        if match:
            groups = match.groups()
            if 'dataset' in pattern:
                symbol = groups[0]
                timeframe = 'H1' if 'H1' in pattern else 'M5'
                return symbol, timeframe, 'Scalping', 'forex'
            elif len(groups) == 4:
                return groups
            elif len(groups) == 3:
                symbol, timeframe, strategy = groups
                market = 'crypto' if symbol in ['BTCUSD', 'ETHUSD'] else 'forex'
                return symbol, timeframe, strategy, market
    return None, None, None, None

def get_timeframe_config(timeframe, market_type, args):
    config = DEFAULT_CONFIGS.get(timeframe, DEFAULT_CONFIGS['M5']).copy()
    
    # Переопределение из аргументов командной строки
    if args.epochs: config['epochs'] = args.epochs
    if args.batch: config['batch_size'] = args.batch
    if args.dropout: config['dropout'] = args.dropout
    if args.l2: config['l2'] = args.l2
    if args.lr: config['learning_rate'] = args.lr
    
    # Корректировка для криптовалют
    if market_type == 'crypto':
        config['threshold'] *= 2.0
        config['dropout'] += 0.1
        config['l2'] *= 1.5
        config['hidden_layers'] = [h * 2 for h in config['hidden_layers']]
        config['epochs'] = int(config['epochs'] * 1.2)
    
    # Корректировка для металлов
    if market_type == 'commodities':
        config['threshold'] *= 1.5
        config['dropout'] += 0.05
        config['epochs'] = int(config['epochs'] * 1.1)
    
    return config

def load_data(x_file, y_file):
    print_color(f"\n📊 Загрузка данных:", Colors.CYAN)
    print(f"   X файл: {x_file}")
    print(f"   Y файл: {y_file}")
    
    try:
        X = pd.read_csv(x_file)
        y = pd.read_csv(y_file)
        print(f"   Записей X: {len(X)}, Y: {len(y)}")
        print(f"   Признаков: {X.shape[1]}")
        
        y_values = y.values.ravel()
        unique, counts = np.unique(y_values, return_counts=True)
        for u, c in zip(unique, counts):
            label_name = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}.get(u, str(u))
            print(f"   {label_name}: {c} ({c/len(y_values)*100:.1f}%)")
        
        return X.values, y_values.astype(int)
    except Exception as e:
        print_color(f"❌ Ошибка: {e}", Colors.RED)
        return None, None

def create_model(input_size, config):
    hidden = config['hidden_layers']
    dropout = config['dropout']
    l2_reg = config['l2']
    
    model = models.Sequential([
        tf.keras.Input(shape=(input_size,)),
        
        layers.Dense(hidden[0], kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout),
        
        layers.Dense(hidden[1], kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout),
        
        layers.Dense(hidden[2], kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.ELU(alpha=1.0),
        layers.Dropout(dropout * 0.8),
        
        layers.Dense(hidden[3], kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('tanh'),
        layers.Dropout(dropout * 0.6),
        
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def export_weights_to_mql4(model, filename):
    weights_list = []
    layer_count = 0
    
    for layer in model.layers:
        # Экспортируем только Dense слои
        if isinstance(layer, tf.keras.layers.Dense):
            layer_weights = layer.get_weights()
            print(f"   Dense слой {layer_count+1}: {len(layer_weights)} массивов")
            for w in layer_weights:
                weights_list.extend(w.flatten())
            layer_count += 1
    
    print(f"   ВСЕГО ВЕСОВ: {len(weights_list)}")
    
    with open(filename, 'w') as f:
        for w in weights_list:
            f.write(f"{w:.10f}\n")
    
    print_color(f"✅ Экспортировано {len(weights_list)} весов в {filename}", Colors.GREEN)

def plot_training_history(history, symbol, timeframe, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title(f'{symbol} {timeframe} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_title(f'{symbol} {timeframe} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print_color(f"✅ График сохранен в {filename}", Colors.GREEN)

def main():
    parser = argparse.ArgumentParser(description='AI Strategy Trainer V9')
    parser.add_argument('--data_x', type=str, required=True, help='X файл (признаки)')
    parser.add_argument('--data_y', type=str, required=True, help='Y файл (метки)')
    parser.add_argument('--market', type=str, default='forex', help='Тип рынка (forex/crypto/commodities)')
    parser.add_argument('--symbol', type=str, help='Символ')
    parser.add_argument('--timeframe', type=str, default='M5', help='Таймфрейм (M5/H1)')
    parser.add_argument('--epochs', type=int, help='Эпохи')
    parser.add_argument('--batch', type=int, help='Батч')
    parser.add_argument('--dropout', type=float, help='Dropout')
    parser.add_argument('--l2', type=float, help='L2 регуляризация')
    parser.add_argument('--lr', type=float, help='Learning rate')
    args = parser.parse_args()
    
    print_color("=" * 70, Colors.CYAN, True)
    print_color("   AI STRATEGY TRAINER V9 - РЕГУЛЯРИЗОВАННАЯ МОДЕЛЬ", Colors.CYAN, True)
    print_color("=" * 70, Colors.CYAN, True)
    
    # Определяем параметры из имени файла если не указаны
    symbol = args.symbol
    timeframe = args.timeframe
    market = args.market
    
    if not symbol:
        symbol, timeframe, strategy, market = extract_info_from_filename(args.data_x)
    
    print_color(f"\n📌 ПАРАМЕТРЫ ОБУЧЕНИЯ:", Colors.CYAN, True)
    print(f"   Символ: {symbol}")
    print(f"   Таймфрейм: {timeframe}")
    print(f"   Тип рынка: {market}")
    
    # Загрузка данных
    X, y = load_data(args.data_x, args.data_y)
    if X is None: return
    
    # Получение конфигурации
    config = get_timeframe_config(timeframe, market, args)
    
    print_color(f"\n📊 КОНФИГУРАЦИЯ МОДЕЛИ:", Colors.CYAN, True)
    print(f"   Архитектура: {config['hidden_layers'][0]}-{config['hidden_layers'][1]}-{config['hidden_layers'][2]}-{config['hidden_layers'][3]}-3")
    print(f"   Эпохи: {config['epochs']}")
    print(f"   Батч: {config['batch_size']}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   L2: {config['l2']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Lookahead: {config['lookahead']} баров")
    print(f"   Порог: {config['threshold']}%")
    
    # Разделение на train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\n   Train: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    scaler_file = f"scaler_{symbol}_{timeframe}_{market}.pkl"
    joblib.dump(scaler, scaler_file)
    print_color(f"✅ Scaler сохранен в {scaler_file}", Colors.GREEN)
    
    # Создание модели
    print_color("\n🧠 Создание модели...", Colors.CYAN)
    model = create_model(X.shape[1], config)
    model.summary()
    print_color(f"\n📊 Всего параметров: {model.count_params():,}", Colors.YELLOW)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            f'best_model_{symbol}_{timeframe}_{market}.keras',
            monitor='val_accuracy', save_best_only=True, verbose=1
        )
    ]
    
    # Обучение
    print_color("\n🚀 Запуск обучения...", Colors.GREEN, True)
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
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
    
    # Сохранение
    model_file = f"model_{symbol}_{timeframe}_{market}.keras"
    weights_file = f"model_{symbol}_{timeframe}_{market}_weights.csv"
    plot_file = f"training_{symbol}_{timeframe}_{market}.png"
    
    model.save(model_file)
    print_color(f"✅ Модель сохранена: {model_file}", Colors.GREEN)
    export_weights_to_mql4(model, weights_file)
    plot_training_history(history, symbol, timeframe, plot_file)
    
    print_color("\n" + "=" * 70, Colors.CYAN, True)
    print_color("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!", Colors.GREEN, True)
    print_color("=" * 70, Colors.CYAN, True)
    print_color(f"\n📌 Для MT4 используйте файл: {weights_file}", Colors.YELLOW)
    
    # Рекомендации
    if val_acc > 0.65:
        print_color(f"\n⚠️ Val Accuracy ({val_acc:.1%}) ВЫСОКАЯ!", Colors.YELLOW)
        print_color("   Рекомендации: --epochs 25-35 --dropout 0.75 --l2 0.02", Colors.YELLOW)
    elif val_acc < 0.52:
        print_color(f"\n⚠️ Val Accuracy ({val_acc:.1%}) НИЗКАЯ!", Colors.YELLOW)
        print_color("   Рекомендации: увеличьте эпохи или уменьшите регуляризацию", Colors.YELLOW)
    else:
        print_color(f"\n✅ Val Accuracy ({val_acc:.1%}) в норме!", Colors.GREEN)

if __name__ == "__main__":
    main()
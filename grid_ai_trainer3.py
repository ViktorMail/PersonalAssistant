#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Strategy Trainer V3 - Адаптивное обучение под разные таймфреймы и стратегии
Поддержка: Scalping (M5/M15), Trend (H1/H4), Grid (любой), Reversal (M15/H1), etc.
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

# Параметры по умолчанию
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32

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

# ============================================================
# КОНФИГУРАЦИИ ДЛЯ РАЗНЫХ ТАЙМФРЕЙМОВ И СТРАТЕГИЙ
# ============================================================

# Настройки для разных таймфреймов
TIMEFRAME_CONFIGS = {
    'M1': {
        'name': 'M1',
        'lookahead_bars': 12,      # 12 минут
        'threshold_pct': 0.01,      # 0.01% движение
        'min_samples': 5000,
        'learning_rate': 0.0008,
        'dropout': 0.35,
        'epochs': 150,
        'batch_size': 64,
        'hidden_layers': [128, 256, 128, 64]
    },
    'M5': {
        'name': 'M5',
        'lookahead_bars': 12,       # 60 минут
        'threshold_pct': 0.05,       # 0.05% движение
        'min_samples': 5000,
        'learning_rate': 0.0008,
        'dropout': 0.3,
        'epochs': 120,
        'batch_size': 64,
        'hidden_layers': [128, 256, 128, 64]
    },
    'M15': {
        'name': 'M15',
        'lookahead_bars': 8,        # 120 минут
        'threshold_pct': 0.08,       # 0.08% движение
        'min_samples': 4000,
        'learning_rate': 0.0007,
        'dropout': 0.3,
        'epochs': 120,
        'batch_size': 64,
        'hidden_layers': [128, 256, 128, 64]
    },
    'M30': {
        'name': 'M30',
        'lookahead_bars': 8,        # 240 минут
        'threshold_pct': 0.1,        # 0.1% движение
        'min_samples': 4000,
        'learning_rate': 0.0007,
        'dropout': 0.25,
        'epochs': 120,
        'batch_size': 64,
        'hidden_layers': [128, 256, 128, 64]
    },
    'H1': {
        'name': 'H1',
        'lookahead_bars': 6,        # 6 часов
        'threshold_pct': 0.15,       # 0.15% движение
        'min_samples': 3000,
        'learning_rate': 0.0006,
        'dropout': 0.25,
        'epochs': 100,
        'batch_size': 32,
        'hidden_layers': [64, 128, 64, 32]
    },
    'H4': {
        'name': 'H4',
        'lookahead_bars': 4,        # 16 часов
        'threshold_pct': 0.25,       # 0.25% движение
        'min_samples': 2000,
        'learning_rate': 0.0005,
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'hidden_layers': [64, 128, 64, 32]
    },
    'D1': {
        'name': 'D1',
        'lookahead_bars': 3,        # 3 дня
        'threshold_pct': 0.5,        # 0.5% движение
        'min_samples': 1000,
        'learning_rate': 0.0005,
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'hidden_layers': [64, 128, 64, 32]
    }
}

# Настройки для разных рынков
MARKET_CONFIGS = {
    'forex': {
        'name': 'Forex',
        'multiplier': 1.0,
        'threshold_scale': 1.0,
        'learning_rate_scale': 1.0,
        'dropout_scale': 1.0
    },
    'crypto': {
        'name': 'Crypto',
        'multiplier': 3.0,
        'threshold_scale': 5.0,      # Крипта требует большего движения
        'learning_rate_scale': 1.2,
        'dropout_scale': 1.2,
        'hidden_layers': [256, 512, 256, 128]  # Для крипты больше нейронов
    },
    'commodities': {
        'name': 'Commodities',
        'multiplier': 1.5,
        'threshold_scale': 2.0,
        'learning_rate_scale': 0.9,
        'dropout_scale': 1.0,
        'hidden_layers': [128, 256, 128, 64]
    }
}

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
            
            if timeframe in TIMEFRAME_CONFIGS:
                if market is None:
                    if symbol in ['BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD', 'ADAUSD', 'DOTUSD']:
                        market = 'crypto'
                    elif symbol in ['XAUUSD', 'XAGUSD', 'CL', 'BRENT']:
                        market = 'commodities'
                    else:
                        market = 'forex'
                return symbol, timeframe, market
    
    return None, None, None

def get_adapted_config(market_type, timeframe):
    """Получить адаптированную конфигурацию для рынка и таймфрейма"""
    
    # Базовые настройки таймфрейма
    config = TIMEFRAME_CONFIGS.get(timeframe, TIMEFRAME_CONFIGS['M5']).copy()
    
    # Настройки рынка
    market_cfg = MARKET_CONFIGS.get(market_type, MARKET_CONFIGS['forex'])
    
    # Адаптируем порог
    config['threshold_pct'] = config['threshold_pct'] * market_cfg['threshold_scale']
    
    # Адаптируем learning rate
    config['learning_rate'] = config['learning_rate'] * market_cfg['learning_rate_scale']
    
    # Адаптируем dropout
    config['dropout'] = min(0.5, config['dropout'] * market_cfg['dropout_scale'])
    
    # Для крипты используем увеличенные скрытые слои
    if market_type == 'crypto':
        config['hidden_layers'] = market_cfg['hidden_layers']
    
    return config

def load_data(filename, market_type, symbol, timeframe):
    """Загрузка данных с правильными параметрами для таймфрейма"""
    print_color(f"\n📊 Загрузка данных для {market_type.upper()} | {symbol} | {timeframe}", Colors.CYAN)
    print(f"   Файл: {filename}")
    
    try:
        data = pd.read_csv(filename)
        print(f"   Записей: {len(data)}")
        
        # Определяем доступные колонки
        feature_cols = []
        for col in ['open', 'high', 'low', 'close', 'volume', 
                    'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'bb_upper', 'bb_middle', 'bb_lower', 'atr',
                    'hour', 'minute', 'day_of_week']:
            if col in data.columns:
                feature_cols.append(col)
        
        print(f"   Используется {len(feature_cols)} признаков")
        X = data[feature_cols].values
        
        # Получаем конфигурацию
        config = get_adapted_config(market_type, timeframe)
        
        # Создаем target с учётом таймфрейма
        if 'target' in data.columns:
            y = data['target'].values
            print(f"   Используется существующий target")
        else:
            print(f"   🎯 Создание target на основе future price...")
            print(f"   Lookahead: {config['lookahead_bars']} баров")
            print(f"   Порог: {config['threshold_pct']}%")
            
            future_returns = data['close'].pct_change(config['lookahead_bars']).shift(-config['lookahead_bars'])
            y = np.zeros(len(data))
            
            y[future_returns > config['threshold_pct']/100] = 0   # BUY
            y[future_returns < -config['threshold_pct']/100] = 1  # SELL
            y[np.abs(future_returns) <= config['threshold_pct']/100] = 2  # HOLD
        
        # Удаляем NaN
        valid_idx = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx].astype(int)
        
        # Статистика
        print(f"\n   📊 Распределение target:")
        print(f"      BUY (0):  {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
        print(f"      SELL (1): {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
        print(f"      HOLD (2): {np.sum(y==2)} ({np.sum(y==2)/len(y)*100:.1f}%)")
        
        return X, y, feature_cols
        
    except Exception as e:
        print_color(f"❌ Ошибка загрузки: {e}", Colors.RED)
        return None, None, None

def create_model(input_size, market_type, timeframe):
    """Создание модели с адаптацией под рынок и таймфрейм"""
    
    config = get_adapted_config(market_type, timeframe)
    hidden = config['hidden_layers']
    dropout_rate = config['dropout']
    l2_lambda = 0.0001
    
    # Для крипты усиливаем регуляризацию
    if market_type == 'crypto':
        l2_lambda = 0.0003
    
    model = models.Sequential([
        tf.keras.Input(shape=(input_size,)),
        
        # Слой 1
        layers.Dense(hidden[0], kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout_rate),
        
        # Слой 2
        layers.Dense(hidden[1], kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout_rate),
        
        # Слой 3
        layers.Dense(hidden[2], kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.BatchNormalization(),
        layers.ELU(alpha=1.0),
        layers.Dropout(dropout_rate * 0.7),
        
        # Слой 4
        layers.Dense(hidden[3], kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.BatchNormalization(),
        layers.Activation('tanh'),
        
        # Выходной слой
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
    parser = argparse.ArgumentParser(description='AI Strategy Trainer V3 - Адаптивное обучение')
    parser.add_argument('--data', type=str, help='CSV файл с данными')
    parser.add_argument('--market', type=str, help='Тип рынка (forex/crypto/commodities)')
    parser.add_argument('--symbol', type=str, help='Торговый символ')
    parser.add_argument('--timeframe', type=str, help='Таймфрейм (M5, M15, H1, H4)')
    parser.add_argument('--epochs', type=int, help='Количество эпох (переопределяет дефолт)')
    parser.add_argument('--batch', type=int, help='Размер батча (переопределяет дефолт)')
    
    args = parser.parse_args()
    
    print_color("=" * 70, Colors.CYAN, True)
    print_color("   AI STRATEGY TRAINER V3 - АДАПТИВНОЕ ОБУЧЕНИЕ", Colors.CYAN, True)
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
        print("   Имя файла должно содержать символ и таймфрейм, например: training_EURUSD_M5.csv")
        return
    
    # Получаем конфигурацию
    config = get_adapted_config(market, timeframe)
    
    # Определяем параметры обучения
    epochs = args.epochs if args.epochs else config['epochs']
    batch_size = args.batch if args.batch else config['batch_size']
    
    print_color(f"\n📊 ПАРАМЕТРЫ ОБУЧЕНИЯ:", Colors.CYAN, True)
    print(f"   Рынок: {market.upper()}")
    print(f"   Символ: {symbol}")
    print(f"   Таймфрейм: {timeframe}")
    print(f"   Файл: {data_file}")
    print(f"   Lookahead: {config['lookahead_bars']} баров")
    print(f"   Порог: {config['threshold_pct']}%")
    print(f"   Эпохи: {epochs}")
    print(f"   Батч: {batch_size}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    
    # Загрузка данных
    X, y, features = load_data(data_file, market, symbol, timeframe)
    if X is None:
        return
    
    # Проверка минимального количества samples
    min_samples = config['min_samples']
    if len(X) < min_samples:
        print_color(f"⚠️ ВНИМАНИЕ: Данных太少 ({len(X)}), рекомендуется минимум {min_samples}", Colors.YELLOW)
    
    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n   Train: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
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
    
    # Подсчёт параметров
    total_params = model.count_params()
    print_color(f"\n📊 Всего параметров: {total_params:,}", Colors.YELLOW)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f'best_model_{symbol}_{timeframe}_{market}.keras', 
                        monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Обучение
    print_color("\n🚀 Запуск обучения...", Colors.GREEN, True)
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
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
        print_color("   ⚠️ Хорошая модель, но требуется доработка", Colors.YELLOW)
    else:
        print_color("   ❌ Модель слабая, соберите больше качественных данных", Colors.RED)

if __name__ == "__main__":
    main()
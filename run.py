#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Strategy Trainer - ИНТЕРАКТИВНЫЙ ЗАПУСК
Поддерживает: M1, M5, M15, M30, H1, H4, D1
Автор: @ViktorKon
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# ===== КОНФИГУРАЦИЯ =====
PROJECT_PATH = r"C:\Projects\PersonalAssistant"

# Все поддерживаемые таймфреймы
TIMEFRAMES = {
    '1': {'name': 'M1',  'desc': '1 минута (скальпинг)', 'epochs': 100},
    '2': {'name': 'M5',  'desc': '5 минут (основной)',   'epochs': 80},
    '3': {'name': 'M15', 'desc': '15 минут',             'epochs': 70},
    '4': {'name': 'M30', 'desc': '30 минут',             'epochs': 60},
    '5': {'name': 'H1',  'desc': '1 час (трендовый)',    'epochs': 60},
    '6': {'name': 'H4',  'desc': '4 часа',               'epochs': 50},
    '7': {'name': 'D1',  'desc': '1 день (долгосрок)',   'epochs': 40},
}

# Рекомендации по параметрам для разных ТФ
RECOMMENDATIONS = {
    'M1':  {'epochs': '25-35', 'batch': '512', 'dropout': '0.8', 'l2': '0.025', 'lr': '0.0001'},
    'M5':  {'epochs': '25-35', 'batch': '256', 'dropout': '0.75', 'l2': '0.02', 'lr': '0.00012'},
    'M15': {'epochs': '22-30', 'batch': '128', 'dropout': '0.7',  'l2': '0.015', 'lr': '0.00014'},
    'M30': {'epochs': '20-28', 'batch': '128', 'dropout': '0.7',  'l2': '0.015', 'lr': '0.00014'},
    'H1':  {'epochs': '20-30', 'batch': '128', 'dropout': '0.7',  'l2': '0.015', 'lr': '0.00015'},
    'H4':  {'epochs': '15-25', 'batch': '64',  'dropout': '0.65', 'l2': '0.012', 'lr': '0.00015'},
    'D1':  {'epochs': '10-20', 'batch': '32',  'dropout': '0.6',  'l2': '0.01',  'lr': '0.0002'},
}

MARKETS = {
    '1': {'name': 'forex',     'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']},
    '2': {'name': 'crypto',    'symbols': ['BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD']},
    '3': {'name': 'commodities','symbols': ['XAUUSD', 'XAGUSD']},
}

def print_header():
    print("=" * 70)
    print("   AI GRID TRAINER - ИНТЕРАКТИВНЫЙ ЗАПУСК")
    print("   Поддержка: M1, M5, M15, M30, H1, H4, D1")
    print("   Автор: @ViktorKon")
    print("=" * 70)
    print()

def print_recommendations(timeframe=None):
    print("\n" + "=" * 70)
    print("   📊 РЕКОМЕНДУЕМЫЕ ЗНАЧЕНИЯ ДЛЯ ДОСТИЖЕНИЯ 55-62% ТОЧНОСТИ")
    print("=" * 70)
    print()
    print("   ╔════════════════╦═════════════════════╦═════════════════════╗")
    print("   ║   Параметр     ║   M5 (основной)     ║   H1 (трендовый)    ║")
    print("   ╠════════════════╬═════════════════════╬═════════════════════╣")
    print("   ║   epochs       ║       25-35         ║       20-30         ║")
    print("   ║   batch        ║       256           ║       128           ║")
    print("   ║   dropout      ║       0.75          ║       0.7           ║")
    print("   ║   l2           ║       0.02          ║       0.015         ║")
    print("   ║   lr           ║       0.00012       ║       0.00015       ║")
    print("   ╚════════════════╩═════════════════════╩═════════════════════╝")
    print()
    print("   💡 При переобучении (Val Accuracy > 65%):")
    print("      - Уменьшите epochs, увеличьте dropout и l2")
    print("   💡 При недообучении (Val Accuracy < 52%):")
    print("      - Увеличьте epochs, уменьшите dropout и l2")
    print()

def select_timeframe():
    print("\n📊 ВЫБЕРИТЕ ТАЙМФРЕЙМ:")
    for key, tf in TIMEFRAMES.items():
        print(f"   {key} - {tf['name']} ({tf['desc']})")
    
    while True:
        choice = input("\nВаш выбор (1-7): ").strip()
        if choice in TIMEFRAMES:
            return TIMEFRAMES[choice]['name']
        print("❌ Неверный выбор, попробуйте еще раз")

def select_market():
    print("\n📊 ВЫБЕРИТЕ ТИП РЫНКА:")
    print("   1 - Forex (EURUSD, GBPUSD, etc.)")
    print("   2 - Crypto (BTCUSD, ETHUSD, etc.)")
    print("   3 - Commodities (XAUUSD, XAGUSD, etc.)")
    
    while True:
        choice = input("\nВаш выбор (1, 2 или 3): ").strip()
        if choice in MARKETS:
            return MARKETS[choice]['name']
        print("❌ Неверный выбор, попробуйте еще раз")

def select_symbol(market_type):
    market = next((m for m in MARKETS.values() if m['name'] == market_type), None)
    if not market:
        return input("Введите символ: ").upper()
    
    print(f"\n💰 ВЫБЕРИТЕ СИМВОЛ ДЛЯ {market_type.upper()}:")
    for i, sym in enumerate(market['symbols'], 1):
        print(f"   {i} - {sym}")
    print(f"   0 - Ввести вручную")
    
    while True:
        choice = input("\nВаш выбор: ").strip()
        if choice == '0':
            return input("Введите символ: ").upper()
        try:
            idx = int(choice)
            if 1 <= idx <= len(market['symbols']):
                return market['symbols'][idx-1]
        except:
            pass
        print("❌ Неверный выбор, попробуйте еще раз")

def find_files(symbol, timeframe):
    """Поиск файлов данных для указанного ТФ и символа"""
    pattern = f"dataset_{timeframe}_{symbol}_{timeframe}"
    
    x_file = Path(f"{pattern}_X.csv")
    y_file = Path(f"{pattern}_Y.csv")
    
    if x_file.exists() and y_file.exists():
        return x_file, y_file
    
    # Проверка в подпапке AI_Data
    ai_data = Path("AI_Data")
    if ai_data.exists():
        x_file = ai_data / f"{pattern}_X.csv"
        y_file = ai_data / f"{pattern}_Y.csv"
        if x_file.exists() and y_file.exists():
            return x_file, y_file
    
    return None, None

def get_user_params(timeframe):
    """Запрос параметров у пользователя"""
    rec = RECOMMENDATIONS.get(timeframe, RECOMMENDATIONS['M5'])
    
    print("\n" + "=" * 70)
    print(f"   ⚙️ НАСТРОЙКИ ОБУЧЕНИЯ ДЛЯ {timeframe}")
    print("=" * 70)
    
    print(f"\n   Рекомендации для {timeframe}:")
    print(f"   ┌─────────────────────────────────────────────────┐")
    print(f"   │  epochs:     {rec['epochs']:<33}│")
    print(f"   │  batch:      {rec['batch']:<33}│")
    print(f"   │  dropout:    {rec['dropout']:<33}│")
    print(f"   │  l2:         {rec['l2']:<33}│")
    print(f"   │  lr:         {rec['lr']:<33}│")
    print(f"   └─────────────────────────────────────────────────┘")
    print(f"\n   (Enter = значения по умолчанию)")
    
    epochs = input(f"\n   Эпохи [{rec['epochs'].split('-')[0]}]: ").strip()
    if not epochs:
        epochs = rec['epochs'].split('-')[0]
    
    batch = input(f"   Батч [{rec['batch']}]: ").strip()
    if not batch:
        batch = rec['batch']
    
    dropout = input(f"   Dropout [{rec['dropout']}]: ").strip()
    if not dropout:
        dropout = rec['dropout']
    
    l2 = input(f"   L2 [{rec['l2']}]: ").strip()
    if not l2:
        l2 = rec['l2']
    
    lr = input(f"   Learning rate [{rec['lr']}]: ").strip()
    if not lr:
        lr = rec['lr']
    
    use_defaults = input(f"\n   Использовать эти параметры? (y/n) [y]: ").strip().lower()
    if use_defaults == 'n':
        return None, None, None, None, None
    
    return epochs, batch, dropout, l2, lr

def auto_run(timeframe, market=None, symbol=None):
    """Автоматический запуск с оптимизированными параметрами"""
    print_header()
    print(f"🤖 АВТОМАТИЧЕСКИЙ РЕЖИМ: {timeframe}")
    
    if not market:
        market = select_market()
    if not symbol:
        symbol = select_symbol(market)
    
    print(f"\n✅ Выбрано: {timeframe} / {market} / {symbol}")
    
    x_file, y_file = find_files(symbol, timeframe)
    if not x_file:
        print(f"\n❌ Файлы не найдены для {symbol} {timeframe}")
        return False
    
    rec = RECOMMENDATIONS.get(timeframe, RECOMMENDATIONS['M5'])
    default_epochs = rec['epochs'].split('-')[0]
    
    cmd = [
        sys.executable, "grid_ai_trainer5.py",
        f"--data_x={x_file}",
        f"--data_y={y_file}",
        f"--market={market}",
        f"--symbol={symbol}",
        f"--timeframe={timeframe}",
        f"--epochs={default_epochs}",
        f"--batch={rec['batch']}",
        f"--dropout={rec['dropout']}",
        f"--l2={rec['l2']}",
        f"--lr={rec['lr']}"
    ]
    
    print(f"\n🚀 Запуск обучения...")
    try:
        subprocess.run(cmd)
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def interactive_run():
    """Интерактивный режим"""
    print_header()
    print_recommendations()
    
    # 1. Выбор таймфрейма
    timeframe = select_timeframe()
    
    # 2. Выбор типа рынка
    market_type = select_market()
    
    # 3. Выбор символа
    symbol = select_symbol(market_type)
    print(f"\n✅ Выбран символ: {symbol}")
    
    # 4. Поиск файлов
    print(f"\n🔍 Поиск файлов для {timeframe} {symbol}...")
    x_file, y_file = find_files(symbol, timeframe)
    
    if not x_file:
        print(f"\n❌ Файлы не найдены!")
        print(f"   Ожидаются: dataset_{timeframe}_{symbol}_{timeframe}_X.csv")
        print(f"   и dataset_{timeframe}_{symbol}_{timeframe}_Y.csv")
        print(f"\n   Сначала соберите данные в MetaTrader!")
        input("\nНажмите Enter для выхода...")
        return
    
    print(f"   ✅ X-файл: {x_file}")
    print(f"   ✅ Y-файл: {y_file}")
    
    # 5. Параметры обучения
    epochs, batch, dropout, l2, lr = get_user_params(timeframe)
    
    if not epochs:
        print("\n❌ Обучение отменено")
        return
    
    # 6. Формирование команды
    cmd = [
        sys.executable, "grid_ai_trainer5.py",
        f"--data_x={x_file}",
        f"--data_y={y_file}",
        f"--market={market_type}",
        f"--symbol={symbol}",
        f"--timeframe={timeframe}",
        f"--epochs={epochs}",
        f"--batch={batch}",
        f"--dropout={dropout}",
        f"--l2={l2}",
        f"--lr={lr}"
    ]
    
    # 7. Запуск
    print("\n" + "=" * 70)
    print(f"🚀 ЗАПУСК ОБУЧЕНИЯ {timeframe} МОДЕЛИ")
    print("=" * 70)
    print(f"   Символ: {symbol}")
    print(f"   Рынок: {market_type}")
    print(f"   Эпохи: {epochs}")
    print(f"   Батч: {batch}")
    print(f"   Dropout: {dropout}")
    print(f"   L2: {l2}")
    print(f"   LR: {lr}")
    print("=" * 70 + "\n")
    
    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
    
    input("\nНажмите Enter для завершения...")

def main():
    parser = argparse.ArgumentParser(description='AI Grid Trainer')
    parser.add_argument('--auto', action='store_true', help='Автоматический режим')
    parser.add_argument('--timeframe', type=str, default='M5', help='Таймфрейм (M1/M5/M15/M30/H1/H4/D1)')
    parser.add_argument('--market', type=str, help='Тип рынка (forex/crypto/commodities)')
    parser.add_argument('--symbol', type=str, help='Символ')
    
    args = parser.parse_args()
    
    os.chdir(PROJECT_PATH)
    
    if args.auto:
        auto_run(args.timeframe, args.market, args.symbol)
    else:
        interactive_run()

if __name__ == "__main__":
    main()
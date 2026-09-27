#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Переобучение модели специально для M5
С пониженным порогом уверенности
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import os

# Отключаем предупреждения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# Параметры для M5
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT = 0.5  # Увеличили dropout для борьбы с переобучением
L2_LAMBDA = 0.001  # Усилили регуляризацию

print("=" * 60)
print("ПЕРЕОБУЧЕНИЕ МОДЕЛИ ДЛЯ M5")
print("=" * 60)

# Загрузка данных
data = pd.read_csv("training_data.csv")
print(f"\n📊 Загружено {len(data)} записей")
print(f"Колонки: {list(data.columns)}")

# Определяем признаки (используем только те, что есть в MT4)
feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 
                'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 
                'bb_lower', 'atr', 'hour', 'minute']

# Проверяем наличие колонок
available_cols = [col for col in feature_cols if col in data.columns]
print(f"\n✅ Используем {len(available_cols)} признаков: {available_cols}")

X = data[available_cols].values

# Создаем target на основе будущих движений (более чувствительный)
future_returns = data['close'].pct_change(3).shift(-3)  # 3 свечи вперед
y = np.zeros(len(data))
y[future_returns > 0.001] = 0  # Buy (0.1%)
y[future_returns < -0.001] = 1  # Sell
y[np.abs(future_returns) <= 0.001] = 2  # Hold

# Удаляем NaN
valid_idx = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
X = X[valid_idx]
y = y[valid_idx].astype(int)

print(f"\n📊 Распределение:")
print(f"   Buy: {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
print(f"   Sell: {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
print(f"   Hold: {np.sum(y==2)} ({np.sum(y==2)/len(y)*100:.1f}%)")

# Разделение
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Нормализация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Сохраняем scaler
joblib.dump(scaler, "scaler_m5.pkl")

# Создаем модель с большим dropout
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(negative_slope=0.01),
    tf.keras.layers.Dropout(DROPOUT),
    
    tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(negative_slope=0.01),
    tf.keras.layers.Dropout(DROPOUT),
    
    tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ELU(alpha=1.0),
    tf.keras.layers.Dropout(DROPOUT * 0.7),
    
    tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(L2_LAMBDA)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('tanh'),
    
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🧠 Архитектура модели:")
model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, 
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                        patience=10, min_lr=1e-6, verbose=1)
]

# Обучение
print("\n🚀 Запуск обучения...")
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Оценка
train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)[1]
val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)[1]

print("\n" + "=" * 60)
print("📈 РЕЗУЛЬТАТЫ")
print("=" * 60)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Val Accuracy:   {val_acc:.4f}")
print(f"Разрыв:         {abs(train_acc - val_acc):.4f}")

# Сохраняем модель и веса
model.save("model_m5.keras")

# Экспорт весов для MT4
weights_list = []
for layer in model.layers:
    if hasattr(layer, 'get_weights'):
        weights = layer.get_weights()
        for w in weights:
            weights_list.extend(w.flatten())

with open("model_m5_weights.csv", 'w') as f:
    for w in weights_list:
        f.write(f"{w:.10f}\n")

print(f"\n✅ Экспортировано {len(weights_list)} весов в model_m5_weights.csv")

# График
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_title('Model Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['accuracy'], label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
ax2.set_title('Model Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_m5.png")
plt.show()

print("\n✅ ГОТОВО! Используйте model_m5_weights.csv в советнике")
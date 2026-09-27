#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grid AI Trainer GUI
Версия: 1.0
Автор: @ViktorKon
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import sys
from datetime import datetime
import queue

# Константы
INPUT_SIZE = 20
HIDDEN1 = 64
HIDDEN2 = 128
HIDDEN3 = 64
HIDDEN4 = 32
OUTPUT_SIZE = 3
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT = 0.3
DEFAULT_L2 = 0.0001

class GridAITrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Grid AI Trainer v1.0")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        
        # Устанавливаем иконку (если есть)
        try:
            self.root.iconbitmap(default='icon.ico')
        except:
            pass
        
        # Переменные для данных
        self.X = None
        self.y = None
        self.feature_names = None
        self.model = None
        self.scaler = None
        self.history = None
        self.training_thread = None
        self.training_running = False
        self.log_queue = queue.Queue()
        
        # Переменные для параметров
        self.epochs_var = tk.IntVar(value=DEFAULT_EPOCHS)
        self.batch_size_var = tk.IntVar(value=DEFAULT_BATCH_SIZE)
        self.learning_rate_var = tk.DoubleVar(value=DEFAULT_LEARNING_RATE)
        self.dropout_var = tk.DoubleVar(value=DEFAULT_DROPOUT)
        self.l2_var = tk.DoubleVar(value=DEFAULT_L2)
        self.test_size_var = tk.DoubleVar(value=0.2)
        
        # Создаем интерфейс
        self.create_menu()
        self.create_main_frame()
        
        # Запускаем обновление лога
        self.update_log()
        
    def create_menu(self):
        """Создание верхнего меню"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Меню File
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load History CSV...", command=self.load_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Load Model Weights...", command=self.load_weights)
        file_menu.add_separator()
        file_menu.add_command(label="Export Weights for MT4", command=self.export_weights)
        file_menu.add_command(label="Export Model (.h5/.keras)", command=self.export_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Меню Training
        train_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Training", menu=train_menu)
        train_menu.add_command(label="Start Training", command=self.start_training, accelerator="F5")
        train_menu.add_command(label="Stop Training", command=self.stop_training, accelerator="F6")
        train_menu.add_separator()
        train_menu.add_command(label="Clear Log", command=self.clear_log)
        
        # Меню View
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Training Graph", command=self.show_graph)
        view_menu.add_command(label="Show Data Statistics", command=self.show_statistics)
        
        # Меню Help
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Author @ViktorKon", command=self.show_author)
        
        # Горячие клавиши
        self.root.bind('<Control-o>', lambda e: self.load_file())
        self.root.bind('<F5>', lambda e: self.start_training())
        self.root.bind('<F6>', lambda e: self.stop_training())
        
    def create_main_frame(self):
        """Создание основного интерфейса"""
        # Основной контейнер
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель (параметры и статистика)
        left_frame = ttk.Frame(main_paned, width=400)
        main_paned.add(left_frame, weight=1)
        
        # Правая панель (лог и график)
        right_frame = ttk.Frame(main_paned, width=800)
        main_paned.add(right_frame, weight=2)
        
        # === ЛЕВАЯ ПАНЕЛЬ ===
        # Заголовок
        title_label = ttk.Label(left_frame, text="GRID AI TRAINER", 
                                font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        # Автор
        author_label = ttk.Label(left_frame, text="Author: @ViktorKon", 
                            font=('Arial', 9))
        author_label.pack()
        
        # Рамка с информацией о файле
        file_frame = ttk.LabelFrame(left_frame, text="Data File", padding=5)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_label = ttk.Label(file_frame, text="No file loaded", 
                                    foreground="red", wraplength=350)
        self.file_label.pack(fill=tk.X)
        
        btn_load = ttk.Button(file_frame, text="Load CSV", command=self.load_file)
        btn_load.pack(pady=5)
        
        # Рамка с параметрами обучения
        params_frame = ttk.LabelFrame(left_frame, text="Training Parameters", padding=5)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Epochs
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, pady=2)
        
        # Batch Size
        ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=1, column=1, pady=2)
        
        # Learning Rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(params_frame, textvariable=self.learning_rate_var, width=10).grid(row=2, column=1, pady=2)
        
        # Dropout
        ttk.Label(params_frame, text="Dropout:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(params_frame, textvariable=self.dropout_var, width=10).grid(row=3, column=1, pady=2)
        
        # L2 Lambda
        ttk.Label(params_frame, text="L2 Lambda:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(params_frame, textvariable=self.l2_var, width=10).grid(row=4, column=1, pady=2)
        
        # Test Size
        ttk.Label(params_frame, text="Test Size:").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Entry(params_frame, textvariable=self.test_size_var, width=10).grid(row=5, column=1, pady=2)
        
        # Рамка с кнопками управления
        control_frame = ttk.LabelFrame(left_frame, text="Control", padding=5)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_start = ttk.Button(control_frame, text="START TRAINING", 
                                command=self.start_training, state=tk.DISABLED)
        self.btn_start.pack(fill=tk.X, pady=2)
        
        self.btn_stop = ttk.Button(control_frame, text="STOP", 
                                    command=self.stop_training, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=2)
        
        self.btn_export = ttk.Button(control_frame, text="Export Weights for MT4", 
                                command=self.export_weights, state=tk.DISABLED)
        self.btn_export.pack(fill=tk.X, pady=2)
        
        # Рамка со статистикой
        stats_frame = ttk.LabelFrame(left_frame, text="Results", padding=5)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=10, width=40, state=tk.DISABLED)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # === ПРАВАЯ ПАНЕЛЬ ===
        # Вкладки
        tab_control = ttk.Notebook(right_frame)
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка с логом
        log_tab = ttk.Frame(tab_control)
        tab_control.add(log_tab, text="Training Log")
        
        self.log_text = scrolledtext.ScrolledText(log_tab, height=20, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вкладка с графиком
        graph_tab = ttk.Frame(tab_control)
        tab_control.add(graph_tab, text="Training Graph")
        
        self.graph_frame = ttk.Frame(graph_tab)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Прогресс-бар
        self.progress = ttk.Progressbar(right_frame, orient=tk.HORIZONTAL, 
                                    length=100, mode='determinate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_label = ttk.Label(right_frame, text="Ready")
        self.progress_label.pack()
        
    def log_message(self, message, level="INFO"):
        """Добавление сообщения в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}\n"
        
        self.log_text.insert(tk.END, formatted_msg)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_log(self):
        """Обновление лога из очереди"""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.update_log)
    
    def clear_log(self):
        """Очистка лога"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("Log cleared")
        
    def load_file(self):
        """Загрузка CSV файла с историей"""
        filename = filedialog.askopenfilename(
            title="Select training data CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.log_message(f"Loading file: {filename}")
            data = pd.read_csv(filename)
            
            # Проверка наличия необходимых колонок
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 
                        'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 
                        'bb_lower', 'atr', 'grid_level', 'distance_to_grid',
                        'avg_grid_distance', 'orders_count', 'hour', 'minute', 'day']
            
            target_col = 'target'
            
            # Проверяем наличие всех колонок
            missing_cols = [col for col in feature_cols if col not in data.columns]
            if missing_cols:
                self.log_message(f"Missing columns: {missing_cols}", "ERROR")
                messagebox.showerror("Error", f"Missing columns: {missing_cols}")
                return
            
            self.X = data[feature_cols].values
            self.y = data[target_col].values
            self.feature_names = feature_cols
            
            # Обновляем информацию
            self.file_label.config(text=f"Loaded: {os.path.basename(filename)}\n"
                                        f"Records: {len(data)}", 
                                foreground="green")
            
            self.log_message(f"Loaded {len(data)} records")
            self.log_message(f"Features: {len(feature_cols)}")
            
            # Показываем статистику
            unique, counts = np.unique(self.y, return_counts=True)
            self.log_message("Target distribution:")
            for u, c in zip(unique, counts):
                name = {0: "BUY", 1: "SELL", 2: "HOLD"}.get(u, f"Class {u}")
                self.log_message(f"  {name}: {c} ({c/len(self.y)*100:.1f}%)")
            
            # Активируем кнопку Start
            self.btn_start.config(state=tk.NORMAL)
            
        except Exception as e:
            self.log_message(f"Error loading file: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def create_model(self):
        """Создание модели нейросети"""
        model = models.Sequential([
            tf.keras.Input(shape=(INPUT_SIZE,)),
            
            layers.Dense(HIDDEN1, kernel_regularizer=regularizers.l2(self.l2_var.get())),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.01),
            layers.Dropout(self.dropout_var.get()),
            
            layers.Dense(HIDDEN2, kernel_regularizer=regularizers.l2(self.l2_var.get())),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.01),
            layers.Dropout(self.dropout_var.get()),
            
            layers.Dense(HIDDEN3, kernel_regularizer=regularizers.l2(self.l2_var.get())),
            layers.BatchNormalization(),
            layers.ELU(alpha=1.0),
            layers.Dropout(self.dropout_var.get() * 0.7),
            
            layers.Dense(HIDDEN4, kernel_regularizer=regularizers.l2(self.l2_var.get())),
            layers.BatchNormalization(),
            layers.Activation('tanh'),
            
            layers.Dense(OUTPUT_SIZE, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate_var.get()),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model_thread(self):
        """Функция обучения в отдельном потоке"""
        try:
            # Разделение данных
            self.log_message("Splitting data...")
            X_train, X_val, y_train, y_val = train_test_split(
                self.X, self.y, test_size=self.test_size_var.get(), 
                random_state=42, stratify=self.y
            )
            
            self.log_message(f"Train set: {len(X_train)} samples")
            self.log_message(f"Validation set: {len(X_val)} samples")
            
            # Нормализация
            self.log_message("Normalizing data...")
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Создание модели
            self.log_message("Creating model...")
            self.model = self.create_model()
            self.model.summary(print_fn=lambda x: self.log_message(x))
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, 
                            restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                patience=5, min_lr=1e-6, verbose=1),
                ModelCheckpoint('best_model_temp.h5', monitor='val_accuracy', 
                            save_best_only=True, verbose=1)
            ]
            
            # Обучение
            self.log_message(f"Starting training for {self.epochs_var.get()} epochs...")
            self.progress_label.config(text="Training in progress...")
            
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, gui):
                    super().__init__()
                    self.gui = gui
                    
                def on_epoch_end(self, epoch, logs=None):
                    if not self.gui.training_running:
                        self.model.stop_training = True
                    
                    progress = (epoch + 1) / self.gui.epochs_var.get() * 100
                    self.gui.progress['value'] = progress
                    
                    msg = (f"Epoch {epoch+1}/{self.gui.epochs_var.get()} - "
                        f"loss: {logs['loss']:.4f} - "
                        f"accuracy: {logs['accuracy']:.4f} - "
                        f"val_loss: {logs['val_loss']:.4f} - "
                        f"val_accuracy: {logs['val_accuracy']:.4f}")
                    self.gui.log_message(msg)
                    self.gui.root.update_idletasks()
            
            callbacks.append(ProgressCallback(self))
            
            self.history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=self.epochs_var.get(),
                batch_size=self.batch_size_var.get(),
                callbacks=callbacks,
                verbose=0
            )
            
            if not self.training_running:
                self.log_message("Training stopped by user")
                self.progress_label.config(text="Training stopped")
            else:
                self.log_message("Training completed successfully!")
                
                # Оценка
                train_loss, train_acc = self.model.evaluate(X_train_scaled, y_train, verbose=0)
                val_loss, val_acc = self.model.evaluate(X_val_scaled, y_val, verbose=0)
                
                results = f"\n=== FINAL RESULTS ===\n"
                results += f"Train Accuracy: {train_acc:.4f}\n"
                results += f"Val Accuracy:   {val_acc:.4f}\n"
                results += f"Train Loss: {train_loss:.4f}\n"
                results += f"Val Loss: {val_loss:.4f}\n"
                
                self.log_message(results)
                
                # Обновляем статистику
                self.update_stats(train_acc, val_acc, train_loss, val_loss)
                
                # Активируем кнопку экспорта
                self.btn_export.config(state=tk.NORMAL)
                
                # Рисуем график
                self.plot_training_history()
                
                self.progress_label.config(text="Training completed!")
            
        except Exception as e:
            self.log_message(f"Training error: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.progress_label.config(text="Training failed")
        
        finally:
            self.training_running = False
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.progress['value'] = 100 if self.training_running else 0
    
    def start_training(self):
        """Запуск обучения"""
        if self.X is None:
            messagebox.showwarning("Warning", "Load data file first!")
            return
        
        if self.training_running:
            messagebox.showwarning("Warning", "Training already in progress")
            return
        
        self.training_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_export.config(state=tk.DISABLED)
        
        self.log_message("=" * 50)
        self.log_message("STARTING TRAINING")
        self.log_message("=" * 50)
        
        self.training_thread = threading.Thread(target=self.train_model_thread)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def stop_training(self):
        """Остановка обучения"""
        if self.training_running:
            self.training_running = False
            self.log_message("Stopping training...")
            self.progress_label.config(text="Stopping...")
    
    def update_stats(self, train_acc, val_acc, train_loss, val_loss):
        """Обновление статистики"""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        stats = f"TRAINING RESULTS\n"
        stats += f"{'='*20}\n\n"
        stats += f"Train Accuracy: {train_acc:.4f}\n"
        stats += f"Val Accuracy:   {val_acc:.4f}\n"
        stats += f"Train Loss: {train_loss:.4f}\n"
        stats += f"Val Loss: {val_loss:.4f}\n\n"
        stats += f"Gap: {abs(train_acc - val_acc):.4f}\n"
        stats += f"{'='*20}\n"
        stats += f"Model: {HIDDEN1}-{HIDDEN2}-{HIDDEN3}-{HIDDEN4}-{OUTPUT_SIZE}"
        
        self.stats_text.insert(1.0, stats)
        self.stats_text.config(state=tk.DISABLED)
    
    def plot_training_history(self):
        """Построение графика обучения"""
        if self.history is None:
            return
        
        # Очищаем фрейм
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # Создаем фигуру
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Встраиваем в tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_graph(self):
        """Показать график в отдельном окне"""
        if self.history is None:
            messagebox.showwarning("Warning", "No training history available")
            return
        
        # Создаем новое окно
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Training History")
        graph_window.geometry("800x400")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_title('Model Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_statistics(self):
        """Показать статистику данных"""
        if self.X is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
        
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Data Statistics")
        stats_window.geometry("500x400")
        
        text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        stats = "=== DATA STATISTICS ===\n\n"
        stats += f"Total samples: {len(self.X)}\n"
        stats += f"Features: {self.X.shape[1]}\n\n"
        
        stats += "Target distribution:\n"
        unique, counts = np.unique(self.y, return_counts=True)
        for u, c in zip(unique, counts):
            name = {0: "BUY", 1: "SELL", 2: "HOLD"}.get(u, f"Class {u}")
            stats += f"  {name}: {c} ({c/len(self.y)*100:.1f}%)\n"
        
        stats += "\nFeature statistics:\n"
        for i, name in enumerate(self.feature_names):
            stats += f"  {name}: mean={np.mean(self.X[:, i]):.4f}, "
            stats += f"std={np.std(self.X[:, i]):.4f}\n"
        
        text.insert(1.0, stats)
        text.config(state=tk.DISABLED)
    
    def export_weights(self):
        """Экспорт весов для MT4"""
        if self.model is None:
            messagebox.showwarning("Warning", "No trained model available")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save weights for MT4",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            weights_list = []
            layer_count = 0
            
            for layer in self.model.layers:
                if hasattr(layer, 'get_weights'):
                    weights = layer.get_weights()
                    if weights:
                        layer_count += 1
                        for w in weights:
                            if len(w.shape) > 1:
                                for row in w:
                                    for val in row:
                                        weights_list.append(float(val))
                            else:
                                for val in w:
                                    weights_list.append(float(val))
            
            with open(filename, 'w') as f:
                for w in weights_list:
                    f.write(f"{w:.10f}\n")
            
            self.log_message(f"Exported {len(weights_list)} weights to {filename}")
            self.log_message(f"Total layers with weights: {layer_count}")
            
            messagebox.showinfo("Success", 
                            f"Weights exported successfully!\n"
                            f"Total values: {len(weights_list)}")
            
        except Exception as e:
            self.log_message(f"Export error: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def export_model(self):
        """Экспорт модели в формате .h5/.keras"""
        if self.model is None:
            messagebox.showwarning("Warning", "No trained model available")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".keras",
            filetypes=[("Keras model", "*.keras"), ("HDF5", "*.h5"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            self.model.save(filename)
            if self.scaler:
                scaler_filename = filename.rsplit('.', 1)[0] + '_scaler.pkl'
                joblib.dump(self.scaler, scaler_filename)
                self.log_message(f"Scaler saved to {scaler_filename}")
            
            self.log_message(f"Model saved to {filename}")
            messagebox.showinfo("Success", "Model exported successfully!")
            
        except Exception as e:
            self.log_message(f"Export error: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def load_weights(self):
        """Загрузка весов модели"""
        filename = filedialog.askopenfilename(
            title="Load model weights",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Здесь можно добавить логику загрузки весов в модель
            self.log_message(f"Loading weights from {filename}")
            messagebox.showinfo("Info", "Weight loading not implemented in this demo")
            
        except Exception as e:
            self.log_message(f"Error loading weights: {str(e)}", "ERROR")
    
    def show_about(self):
        """О программе"""
        about_text = """Grid AI Trainer v1.0
        
Программа для обучения нейросети для сеточного советника MT4.

Features:
• Загрузка CSV файлов с историческими данными
• Настройка параметров обучения
• Визуализация процесса обучения
• Экспорт весов для MT4
• Многопоточное обучение

Requirements:
• TensorFlow 2.x
• Python 3.8+
• tkinter (встроен)

© 2024 @ViktorKon"""
        
        messagebox.showinfo("About", about_text)
    
    def show_author(self):
        """Информация об авторе"""
        author_text = """Author: @ViktorKon

Telegram: @ViktorKon
Email: viktorkon@example.com"""
        
        messagebox.showinfo("Author", author_text)

def main():
    root = tk.Tk()
    app = GridAITrainerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
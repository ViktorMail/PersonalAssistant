import sys
import subprocess

print("=" * 50)
print("ДИАГНОСТИКА ОКРУЖЕНИЯ PYTHON")
print("=" * 50)
print(f"Текущая директория: {__file__}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("=" * 50)

# Проверка установленных пакетов
print("\nУСТАНОВЛЕННЫЕ ПАКЕТЫ:")
result = subprocess.run([sys.executable, "-m", "pip", "list"], 
    capture_output=True, text=True)
print(result.stdout)

print("\n" + "=" * 50)
print("ТЕСТИРОВАНИЕ ИМПОРТОВ")
print("=" * 50)

# Тест 1: tensorflow
try:
    import tensorflow as tf
    print(f"✅ tensorflow {tf.__version__}")
    print(f"   путь: {tf.__file__}")
except ImportError as e:
    print(f"❌ tensorflow: {e}")

# Тест 2: keras из tensorflow
try:
    from tensorflow import tf
    print(f"✅ tensorflow.keras {tf.keras.__version__}")
    print(f"   путь: {tf.keras.__file__}")
except ImportError as e:
    print(f"❌ tensorflow.keras: {e}")

# Тест 3: layers
try:
    from tensorflow.keras import layers
    print("✅ tensorflow.keras.layers")
except ImportError as e:
    print(f"❌ tensorflow.keras.layers: {e}")

# Тест 4: regularizers
try:
    from tensorflow.keras import regularizers
    print("✅ tensorflow.keras.regularizers")
except ImportError as e:
    print(f"❌ tensorflow.keras.regularizers: {e}")

# Тест 5: models
try:
    from tensorflow.keras import models
    print("✅ tensorflow.keras.models")
except ImportError as e:
    print(f"❌ tensorflow.keras.models: {e}")

# Тест 6: optimizers
try:
    from tensorflow.keras.optimizers import Adam
    print("✅ tensorflow.keras.optimizers.Adam")
except ImportError as e:
    print(f"❌ tensorflow.keras.optimizers.Adam: {e}")

# Тест 7: callbacks
try:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    print("✅ tensorflow.keras.callbacks")
except ImportError as e:
    print(f"❌ tensorflow.keras.callbacks: {e}")

print("=" * 50)
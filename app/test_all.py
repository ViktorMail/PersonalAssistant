import sys
print(f"Python: {sys.executable}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn: {e}")

try:
    import matplotlib
    print(f"✅ Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Matplotlib: {e}")

try:
    import joblib
    print(f"✅ Joblib: {joblib.__version__}")
except ImportError as e:
    print(f"❌ Joblib: {e}")
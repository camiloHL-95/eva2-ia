# train_diabetes.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
OUT_DIR = BASE_DIR / "models"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Leer dataset y normalizar columnas
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

if "outcome" not in df.columns:
    raise ValueError(f"❌ La columna 'outcome' no fue encontrada. Columnas disponibles: {df.columns.tolist()}")

X = df.drop("outcome", axis=1)
y = df["outcome"]

# Pipeline de preprocesamiento y modelo
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipe.fit(X, y)

# Guardar modelo
out_path = OUT_DIR / "diabetes.pkl"
joblib.dump(pipe, out_path)
print(f"✅ Modelo de diabetes guardado en {out_path}")

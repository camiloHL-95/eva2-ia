import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "insurance.csv"
OUT_DIR = BASE_DIR / "models"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Cargar y normalizar columnas
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

# Chequeo básico de columnas esperadas
expected = {"age","sex","bmi","children","smoker","region","charges"}
missing = expected - set(df.columns)
if missing:
    raise ValueError(f"❌ Faltan columnas en insurance.csv: {missing}. Encontradas: {df.columns.tolist()}")

# 2) Asegurar tipos (por si vienen como string)
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
df["children"] = pd.to_numeric(df["children"], errors="coerce")

# limpieza
for c in ["sex","smoker","region"]:
    df[c] = df[c].astype(str).str.strip().str.lower()

# Eliminar filas con NaN en las columnas clave
df = df.dropna(subset=["age","bmi","children","sex","smoker","region","charges"]).reset_index(drop=True)

# 3) Split X/y
X = df.drop("charges", axis=1)
y = df["charges"]

# 4) Preprocesamiento + modelo
cat_cols = ["sex", "smoker", "region"]
num_cols = ["age", "bmi", "children"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ],
    remainder="drop"
)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

# 5) Entrenar (puedes añadir train_test_split si quieres evaluar)
pipe.fit(X, y)

# 6) Guardar
out_path = OUT_DIR / "insurance.pkl"
joblib.dump(pipe, out_path)
print(f"✅ Modelo de seguro guardado en {out_path}")

# 🧾 Medical Insurance & Diabetes Prediction API

## 📘 Descripción general
Este proyecto implementa dos modelos de *Machine Learning*:

1. **Predicción de costos de seguro médico** (Regresión Lineal).  
2. **Predicción de diabetes** (Regresión Logística).  

Ambos modelos fueron entrenados y desplegados como una **API REST con FastAPI**, permitiendo enviar datos en formato JSON y recibir predicciones inmediatas.

---

## 🧩 Modelos utilizados
- **Seguro médico:** [Medical Insurance Cost with Linear Regression (Kaggle)](https://www.kaggle.com/code/mariapushkareva/medical-insurance-cost-with-linear-regression)  
- **Diabetes:** [Diabetes Logistic Regression (Kaggle)](https://www.kaggle.com/code/arezalo/diabetes-logistic-regression)

---

## 🧠 1) Umbral ideal para el modelo de predicción de diabetes
El modelo de **regresión logística** entrega una probabilidad de que el paciente tenga diabetes (`predict_proba`).  
Para decidir el diagnóstico (0 = no, 1 = sí), se definió el **umbral óptimo mediante la estadística de Youden**:

\[
J = \text{Sensibilidad} + \text{Especificidad} - 1
\]

El punto que maximiza `J` define el equilibrio entre sensibilidad y especificidad.  
En este caso, el **umbral ideal fue ≈ 0.32**, lo que significa que si la probabilidad predicha ≥ 0.32, el modelo clasifica el caso como positivo.

> 🔍 Este valor permite detectar más casos de diabetes sin elevar excesivamente los falsos positivos.

---

## 💸 2) Factores que más influyen en los costos del seguro médico
Según el modelo de regresión lineal y el análisis con *RandomForestRegressor*:

| Factor | Influencia estimada |
|--------|---------------------|
| **smoker** | 🚬 Muy alta (fumadores pagan mucho más) |
| **age** | 📈 Aumenta el costo con la edad |
| **bmi** | 🧍‍♂️ Elevado índice de masa corporal aumenta riesgo |
| **region** | 🌎 Afecta ligeramente según localización |
| **children** | 👶 Incremento leve por número de dependientes |
| **sex** | ⚧ Influencia baja o marginal |

> 🧾 Estos resultados son consistentes con el dataset: los fumadores y personas con mayor IMC presentan los costos más altos.

---

## ⚖️ 3) Análisis comparativo de características (RandomForest)
Se entrenó un **RandomForestClassifier (diabetes)** y un **RandomForestRegressor (seguros)**, calculando importancias normalizadas.  
Comparativamente:

| Variable | Diabetes (importancia) | Seguro médico (importancia) |
|-----------|------------------------|------------------------------|
| `bmi` | Alta | Alta |
| `age` | Alta | Alta |
| `smoker` | — | Muy alta |
| `glucose` | Muy alta | — |
| `bloodpressure` | Moderada | — |

📊 **Conclusión:**  
Variables como `age` y `bmi` influyen en ambos modelos, lo que sugiere correlaciones entre salud metabólica y costos médicos.  
En cambio, `glucose` y `smoker` son específicas de cada contexto.

---

## ⚙️ 4) Técnicas de optimización aplicadas

| Técnica | Aplicación | Beneficio |
|----------|-------------|------------|
| **Pipeline + Preprocesamiento** | Integración de escalado y codificación dentro del modelo (`StandardScaler`, `OneHotEncoder`) | Evita fugas de datos y mejora estabilidad |
| **Regularización (L2)** | En regresión logística | Previene sobreajuste |
| **Hyperparameter tuning** | `RandomizedSearchCV` o `Optuna` (opcional) | Mejora el AUC o R² |
| **Feature engineering** | Variables cruzadas (`bmi*age`, `smoker*bmi`) | Incrementa la capacidad predictiva |
| **Balance de clases (SMOTE o class_weight)** | En dataset de diabetes | Mejora sensibilidad |

> ⚙️ En pruebas, la combinación de *regularización + validación cruzada* aumentó el rendimiento general y redujo el sesgo.

---

## 🧾 5) Contexto de los datos

### 🧬 *Diabetes dataset (Pima Indians)*
Contiene 768 registros con variables clínicas:
- Glucosa, presión, insulina, BMI, edad, etc.  
- Etiqueta: `outcome` (1 = diabetes, 0 = no).

Procede de un estudio médico en mujeres de origen Pima, por lo que **no representa toda la población mundial**.

### 🏥 *Insurance dataset*
1.338 registros de pólizas de salud con información demográfica:
- `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`.  
- Variables categóricas y numéricas combinadas.

Ambos son **datasets públicos de Kaggle**, ideales para fines educativos y de comparación de modelos.

---

## ⚖️ 6) Sesgo y limitaciones de los modelos

| Tipo de sesgo | Explicación | Impacto |
|----------------|-------------|----------|
| **Sesgo de muestreo** | Los datos provienen de poblaciones específicas (p. ej. comunidad Pima o usuarios de EE.UU.) | Los resultados no generalizan globalmente |
| **Sesgo de variables omitidas** | Falta información socioeconómica o clínica más profunda | Se simplifica el fenómeno real |
| **Desbalance de clases (diabetes)** | Menor cantidad de casos positivos | Afecta precisión sin ajuste de umbral |
| **Sesgo de correlación** | Algunas variables actúan como proxies (ej. `region` ↔ nivel de vida) | Riesgo de interpretaciones erróneas |

> 💡 Para mitigarlo se aplicaron técnicas de escalado, validación cruzada y regularización. Sin embargo, los modelos deben considerarse demostrativos y no clínicos.

---

## 🚀 Ejecución de la aplicación

### 1️⃣ Crear entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate   # En Linux/Mac
.venv\Scripts\activate      # En Windows
pip install -r requirements.txt
```

### 2️⃣ Entrenar modelos
```bash
python train_insurance.py
python train_diabetes.py
```

Generará:
```
models/
 ├─ insurance.pkl
 └─ diabetes.pkl
```

### 3️⃣ Levantar API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Accede a la documentación interactiva en:  
👉 **http://localhost:8000/docs**

---

## 📦 Ejemplo de uso

### `/predict/insurance`
```json
{
  "age": 42,
  "sex": "male",
  "bmi": 27.5,
  "children": 2,
  "smoker": "no",
  "region": "southeast"
}
```
**Respuesta:**
```json
{ "predicted_cost": 7961.91 }
```

### `/predict/diabetes?threshold=0.32`
```json
{
  "Pregnancies": 2,
  "Glucose": 145,
  "BloodPressure": 80,
  "SkinThickness": 30,
  "Insulin": 100,
  "BMI": 33.2,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 45
}
```
**Respuesta:**
```json
{ "probability": 0.68, "threshold_used": 0.32, "prediction": 1 }
```

---

## 🧱 Estructura del repositorio
```
Proyecto-app-ia-dos/
├─ data/
│  ├─ diabetes.csv
│  └─ insurance.csv
├─ models/
│  ├─ diabetes.pkl
│  └─ insurance.pkl
├─ main.py
├─ train_diabetes.py
├─ train_insurance.py
├─ requirements.txt
└─ README.md
```

---

## 👤 Autor
**Camilo Herrera - Victor Mardones - Debora Leal**  
Ingeniería en Informática — INACAP  

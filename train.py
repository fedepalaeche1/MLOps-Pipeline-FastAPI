"""
Entrena y guarda el pipeline de aprobación de crédito (Taller 1 - Punto 1).

El pipeline integra en un solo objeto:
  - Preprocesamiento numérico:   SimpleImputer(mean)  -> StandardScaler
  - Preprocesamiento categórico: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown="ignore")
  - Modelo: LogisticRegression(C=0.1, max_iter=1000)
    (mejores hiperparámetros encontrados con GridSearchCV en el Taller 1)

Uso:
    python train.py
Salida:
    data/credito.csv      -> dataset utilizado
    pipeline.joblib       -> pipeline completo (preprocesamiento + modelo)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "credito.csv"
PIPELINE_PATH = BASE_DIR / "pipeline.joblib"

NUM_COLS = ["edad", "ingreso_mensual", "deuda_actual", "antiguedad_laboral"]
CAT_COLS = ["tipo_contrato", "ciudad", "nivel_educativo"]
TARGET = "aprobado"

# Mejores hiperparámetros del GridSearchCV del Taller 1
BEST_PARAMS = {
    "num_imputer_strategy": "mean",
    "C": 0.1,
    "max_iter": 1000,
}


def generar_datos(n: int = 1500) -> pd.DataFrame:
    """Dataset sintético de aprobación de crédito (mismo que en el Taller 1)."""
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "edad": rng.integers(18, 75, n).astype(float),
        "ingreso_mensual": rng.lognormal(mean=8.0, sigma=0.5, size=n).round(0),
        "deuda_actual": rng.gamma(shape=2.0, scale=1500, size=n).round(0),
        "antiguedad_laboral": rng.integers(0, 30, n).astype(float),
        "tipo_contrato": rng.choice(["indefinido", "fijo", "independiente"], n, p=[0.5, 0.3, 0.2]),
        "ciudad": rng.choice(["Medellin", "Bogota", "Cali", "Barranquilla"], n),
        "nivel_educativo": rng.choice(["bachiller", "tecnico", "profesional", "posgrado"], n),
    })
    score = (
        0.00025 * df["ingreso_mensual"]
        - 0.0004 * df["deuda_actual"]
        + 0.06 * df["antiguedad_laboral"]
        + df["tipo_contrato"].map({"indefinido": 0.8, "fijo": 0.2, "independiente": -0.4})
        + df["nivel_educativo"].map({"bachiller": -0.3, "tecnico": 0.0, "profesional": 0.3, "posgrado": 0.5})
        + rng.normal(0, 0.8, n)
    )
    df[TARGET] = (score > np.median(score)).astype(int)

    # ~8% de valores faltantes en algunas columnas
    for col in ["ingreso_mensual", "antiguedad_laboral", "tipo_contrato", "nivel_educativo"]:
        df.loc[rng.random(n) < 0.08, col] = np.nan
    return df


def construir_pipeline() -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=BEST_PARAMS["num_imputer_strategy"])),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUM_COLS),
        ("cat", categorical_pipe, CAT_COLS),
    ])
    return Pipeline([
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(C=BEST_PARAMS["C"], max_iter=BEST_PARAMS["max_iter"])),
    ])


def main() -> None:
    df = generar_datos()
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset guardado en {DATA_PATH.name}: {df.shape[0]} filas, {df.shape[1]} columnas")

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nPipeline entrenado:")
    print(pipeline)
    print(f"\nAccuracy en test: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"Pipeline completo guardado en: {PIPELINE_PATH.name}")

    # Verificación: cargar el artefacto y predecir sobre datos crudos
    cargado = joblib.load(PIPELINE_PATH)
    ejemplo = X_test.head(3)
    print("\nVerificación con el pipeline cargado (datos crudos):")
    print(ejemplo.assign(prediccion=cargado.predict(ejemplo),
                         prob_aprobado=cargado.predict_proba(ejemplo)[:, 1].round(3)))


if __name__ == "__main__":
    main()

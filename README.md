# MLOps-Pipeline-FastAPI

API de predicción de **aprobación de crédito** construida con FastAPI. Sirve un `Pipeline` de scikit-learn
(Taller 1) que integra el preprocesamiento y el modelo en un único artefacto (`pipeline.joblib`).

**Autor:** Federico Palacio Echeverri · **Docente:** Santiago Mendez Mejia<br>
Universidad EIA · MLOps · Taller 3 (2026)

Repositorio: https://github.com/fedepalaeche1/MLOps-Pipeline-FastAPI

## Pipeline

| Paso | Columnas | Transformación |
|---|---|---|
| Numéricas | `edad`, `ingreso_mensual`, `deuda_actual`, `antiguedad_laboral` | `SimpleImputer(strategy="mean")` → `StandardScaler` |
| Categóricas | `tipo_contrato`, `ciudad`, `nivel_educativo` | `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")` |
| Modelo | — | `LogisticRegression(C=0.1, max_iter=1000)` |

Los hiperparámetros (`C=0.1`, imputación numérica `mean`) son los mejores encontrados con `GridSearchCV`
en el Taller 1. Accuracy en test: **0.803**.

Como el pipeline imputa los faltantes y usa `handle_unknown="ignore"`, la API acepta valores nulos y ciudades
que no aparecieron en el entrenamiento.

## Estructura

```
├── train.py                  # genera los datos, entrena y guarda el pipeline
├── pipeline.joblib           # pipeline entrenado (preprocesamiento + modelo)
├── data/credito.csv          # dataset utilizado
├── main.py                   # API FastAPI
├── request_example.py        # ejemplo de solicitud de predicción
├── requirements.txt          # dependencias
└── deploy/pipeline-api.service  # servicio systemd para EC2
```

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Mensaje de bienvenida |
| GET | `/health` | Estado del servicio y modelo cargado |
| POST | `/predict` | Predicción para una o varias solicitudes |
| GET | `/docs` | Documentación interactiva (Swagger UI) |

Ejemplo de solicitud:

```json
{
  "solicitudes": [
    {"edad": 35, "ingreso_mensual": 6000, "deuda_actual": 1000, "antiguedad_laboral": 8,
     "tipo_contrato": "indefinido", "ciudad": "Medellin", "nivel_educativo": "profesional"}
  ]
}
```

Respuesta:

```json
{"predicciones": [{"aprobado": 1, "resultado": "APROBADO", "prob_aprobado": 0.9284}]}
```

## Ejecución local

Requiere Python 3.11 o superior.

```bash
git clone https://github.com/fedepalaeche1/MLOps-Pipeline-FastAPI.git
cd MLOps-Pipeline-FastAPI
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python train.py                          # opcional: reentrena y regenera pipeline.joblib
uvicorn main:app --host 127.0.0.1 --port 8000
```

En otra terminal:

```bash
python request_example.py
# o
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"solicitudes":[{"edad":35,"ingreso_mensual":6000,"deuda_actual":1000,"antiguedad_laboral":8,"tipo_contrato":"indefinido","ciudad":"Medellin","nivel_educativo":"profesional"}]}'
```

También se puede probar desde `http://127.0.0.1:8000/docs`.

## Despliegue en AWS EC2 (Ubuntu)

1. En el grupo de seguridad de la instancia, abrir **SSH (22)** y **TCP 8000**.
2. Instalar dependencias y clonar el repositorio:

```bash
sudo apt update && sudo apt install -y git python3-venv
git clone https://github.com/fedepalaeche1/MLOps-Pipeline-FastAPI.git
cd MLOps-Pipeline-FastAPI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Probar manualmente:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
# desde el navegador: http://<IP_PUBLICA>:8000/docs
```

### Servicio systemd

Para que la API corra en segundo plano, arranque con la instancia y se reinicie si falla:

```bash
sudo cp deploy/pipeline-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pipeline-api
sudo systemctl status pipeline-api
journalctl -u pipeline-api -f          # logs
```

El archivo asume que el repositorio está en `/home/ubuntu/MLOps-Pipeline-FastAPI` y que el usuario es `ubuntu`.
Si cambia la ruta, ajuste `WorkingDirectory`, `Environment` y `ExecStart`.

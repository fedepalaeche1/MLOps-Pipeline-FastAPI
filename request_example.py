"""
Ejemplo de solicitud de predicción a la API.

Uso:
    python request_example.py                          # API local (http://127.0.0.1:8000)
    python request_example.py http://<IP_PUBLICA>:8000  # API en EC2
"""
import json
import sys

import requests

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")

payload = {
    "solicitudes": [
        {"edad": 35, "ingreso_mensual": 6000, "deuda_actual": 1000, "antiguedad_laboral": 8,
         "tipo_contrato": "indefinido", "ciudad": "Medellin", "nivel_educativo": "profesional"},
        # Valores faltantes y una ciudad no vista en entrenamiento: el pipeline los maneja
        {"edad": 22, "ingreso_mensual": None, "deuda_actual": 9000, "antiguedad_laboral": 0,
         "tipo_contrato": "independiente", "ciudad": "Pereira", "nivel_educativo": None},
    ]
}

print(f"POST {URL}/predict")
print("Solicitud:")
print(json.dumps(payload, indent=2, ensure_ascii=False))

response = requests.post(f"{URL}/predict", json=payload, timeout=10)
print(f"\nStatus: {response.status_code}")
print("Respuesta:")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

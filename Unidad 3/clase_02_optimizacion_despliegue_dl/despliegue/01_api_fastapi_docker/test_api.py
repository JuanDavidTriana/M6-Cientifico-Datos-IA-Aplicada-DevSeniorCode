"""
Cliente de prueba: consulta /health y envía una imagen de Fashion-MNIST a /predict.

    python test_api.py
    python test_api.py --host http://localhost:8000 --api-key MI_CLAVE
"""
import argparse
import os

import numpy as np
import requests

MUESTRA = os.path.join(os.path.dirname(__file__), "..", "..", "datos", "fashion_mnist_muestra.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:8000")
    p.add_argument("--api-key", default=os.environ.get("API_KEY"))
    args = p.parse_args()

    cabeceras = {"X-API-Key": args.api_key} if args.api_key else {}
    print("GET /health ->", requests.get(f"{args.host}/health", timeout=10).json())

    datos = np.load(MUESTRA)
    imagen, etiqueta = datos["x"][0], int(datos["y"][0])
    r = requests.post(f"{args.host}/predict", json={"imagen": imagen.tolist()},
                      headers=cabeceras, timeout=10)
    print("POST /predict ->", r.status_code, r.json(), "| etiqueta real:", etiqueta)


if __name__ == "__main__":
    main()

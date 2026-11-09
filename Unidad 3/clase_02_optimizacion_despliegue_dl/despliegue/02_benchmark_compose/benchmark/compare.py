"""
Compara en vivo los dos servicios levantados con docker compose:

    modelo-base       -> http://localhost:8001
    modelo-optimizado -> http://localhost:8002

Envía las MISMAS imágenes reales de Fashion-MNIST (datos/fashion_mnist_muestra.npz)
a ambos y mide latencia (extremo a extremo), throughput y accuracy.

    python benchmark/compare.py            # 100 peticiones por servicio
    python benchmark/compare.py --n 300
"""
import argparse
import os
import time

import numpy as np
import requests

BASE_URL = "http://localhost:8001"
OPT_URL = "http://localhost:8002"

AQUI = os.path.dirname(__file__)
MUESTRA = os.path.join(AQUI, "..", "..", "..", "datos", "fashion_mnist_muestra.npz")
MODEL_BASE = os.path.join(AQUI, "..", "servicio_baseline", "model", "modelo_base.h5")
MODEL_OPT = os.path.join(AQUI, "..", "servicio_optimizado", "model", "modelo_optimizado.tflite")
REPORTE = os.path.join(AQUI, "reporte.md")


def esperar_servicio(url, intentos=30, espera=2):
    for _ in range(intentos):
        try:
            r = requests.get(f"{url}/health", timeout=3)
            if r.status_code == 200 and r.json().get("modelo_cargado"):
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(espera)
    return False


def medir(url, x, y):
    latencias, aciertos = [], 0
    t0 = time.perf_counter()
    for img, etiqueta in zip(x, y):
        t = time.perf_counter()
        r = requests.post(f"{url}/predict", json={"imagen": img.tolist()}, timeout=30)
        r.raise_for_status()
        latencias.append((time.perf_counter() - t) * 1000)
        aciertos += int(r.json()["clase_id"] == int(etiqueta))
    total = time.perf_counter() - t0
    lat = np.array(latencias)
    return {
        "lat_media_ms": lat.mean(),
        "lat_p95_ms": np.percentile(lat, 95),
        "throughput": len(x) / total,
        "accuracy": aciertos / len(x),
    }


def mb(path):
    return os.path.getsize(path) / 1024 / 1024 if os.path.exists(path) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="Peticiones por servicio")
    args = p.parse_args()

    print("Esperando a que ambos servicios estén listos...")
    if not (esperar_servicio(BASE_URL) and esperar_servicio(OPT_URL)):
        print("Algún servicio no respondió. ¿Corriste 'docker compose up --build'?")
        return

    datos = np.load(MUESTRA)
    x, y = datos["x"][: args.n], datos["y"][: args.n]

    print(f"Midiendo BASE (8001) con {len(x)} imágenes...")
    b = medir(BASE_URL, x, y)
    print(f"Midiendo OPTIMIZADO (8002) con {len(x)} imágenes...")
    o = medir(OPT_URL, x, y)
    sb, so = mb(MODEL_BASE), mb(MODEL_OPT)

    filas = [
        ("Tamaño del modelo (MB)", sb, so),
        ("Latencia promedio (ms)", b["lat_media_ms"], o["lat_media_ms"]),
        ("Latencia p95 (ms)", b["lat_p95_ms"], o["lat_p95_ms"]),
        ("Throughput (req/s)", b["throughput"], o["throughput"]),
        ("Accuracy en la muestra", b["accuracy"], o["accuracy"]),
    ]
    print(f"\n{'Métrica':28}{'Base':>12}{'Optimizado':>14}")
    for nombre, vb, vo in filas:
        print(f"{nombre:28}{vb:>12.3f}{vo:>14.3f}")

    with open(REPORTE, "w", encoding="utf-8") as f:
        f.write("# Reporte de benchmark — base vs optimizado\n\n")
        f.write(f"Peticiones por servicio: {len(x)}\n\n| Métrica | Base | Optimizado |\n|---|---|---|\n")
        for nombre, vb, vo in filas:
            f.write(f"| {nombre} | {vb:.3f} | {vo:.3f} |\n")
    print(f"\nReporte guardado en {REPORTE}")


if __name__ == "__main__":
    main()

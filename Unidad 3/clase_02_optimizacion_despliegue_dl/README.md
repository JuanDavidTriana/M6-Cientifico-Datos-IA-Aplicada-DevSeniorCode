# Unidad 3 · Clase 2: Optimización y **despliegue** de modelos de Deep Learning

**Módulo:** Científico de Datos e Inteligencia Artificial Aplicada · **Unidad 3:** IA Aplicada, Databricks y Arquitecturas Modernas

**Caso:** el clasificador de prendas de la tienda (Fashion-MNIST). **Hoy no se entrena nada**: el modelo llega ya entrenado, podado y cuantizado, y el trabajo es medirlo, empaquetarlo como servicio y dejarlo corriendo.

```
modelo.h5 ─► medir ─► TFLite ─► API (FastAPI) ─► Docker ─► servidor (SSH) ─► benchmark
```

| Parte | Pregunta | Técnicas / herramientas |
|---|---|---|
| **Medir** | ¿Cuánto pesa, qué tan rápido y qué tan preciso es cada artefacto? | tamaño, accuracy, latencia por imagen |
| **Optimización (verificar)** | ¿Hicieron su trabajo el pruning y la quantization? | % de pesos en cero, conversión a TFLite (dinámica / float16) |
| **Servir** | ¿Cómo lo consumen otros sistemas? | FastAPI, Pydantic, `/health`, carga única del modelo |
| **Configurar** | ¿Dónde viven rutas y secretos? | `.env`, `API_KEY`, `.env.example` |
| **Empaquetar** | ¿Cómo corre igual en cualquier máquina? | Dockerfile, caché de capas, `HEALTHCHECK`, `docker compose` |
| **Llegar al servidor** | ¿Cómo se accede de forma segura? | SSH con llaves, `scp`, `~/.ssh/config` |

## 🗂️ Contenido de la carpeta

| Archivo | Para quién | Descripción |
|---|---|---|
| `notebooks/Clase02_Despliegue_DL_DOCENTE.ipynb` | Docente | Notebook **completo y ejecutado**, con guía de tiempos y notas por sección |
| `notebooks_estudiantes/Clase02_Despliegue_DL_ESTUDIANTE.ipynb` | Estudiantes | Mismo notebook con las celdas de código clave **vacías** (y una pista) para live coding; las celdas de apoyo vienen listas |
| `web/guia-clase-02-despliegue.html` | Docente | Guía para dar la clase: agenda, "dilo así", comandos de terminal y qué hacer si algo falla |
| `modelos/` | Ambos | Los 3 artefactos entregados: `modelo_base.h5`, `modelo_podado.h5`, `modelo_optimizado.tflite` |
| `datos/fashion_mnist_muestra.npz` | Ambos | 1 000 imágenes reales de test (**no requiere internet**) |
| `despliegue/01_api_fastapi_docker/` | Ambos | Proyecto de la API: `app/`, `Dockerfile`, `.env.example`, `.dockerignore`, `test_api.py` |
| `despliegue/02_benchmark_compose/` | Ambos | `docker-compose.yml` con dos servicios (base vs optimizado) y `benchmark/compare.py` |

## ⏱️ Agenda (≈ 4 horas)

| Tiempo | Sección |
|---|---|
| 0:00 – 0:20 | 0–1. Setup y del notebook a producción |
| 0:20 – 0:55 | 2. Medir antes de decidir |
| 0:55 – 1:15 | 3. Qué hicieron pruning y quantization (sin reentrenar) |
| 1:15 – 1:25 | ☕ Pausa |
| 1:25 – 2:15 | 4. API con FastAPI (live coding) |
| 2:15 – 2:40 | 5. `.env` y secretos |
| 2:40 – 3:20 | 6–7. Docker y docker compose (terminal) |
| 3:20 – 3:40 | 8. SSH |
| 3:40 – 4:00 | 9. Checklist, retos y cierre |

## ⚙️ Preparación

```bash
pip install -r requirements.txt
```

- El notebook corre en **CPU** y no necesita internet ni GPU.
- Para las secciones 6–7 se necesita **Docker** (Docker Desktop). El primer `docker build` descarga TensorFlow y tarda varios minutos: conviene lanzarlo al empezar la sección 4.
- Para la sección 8 se necesita `ssh-keygen` (viene con Git for Windows, macOS y Linux).

## 🎯 Objetivos
1. Medir tamaño, latencia y accuracy de un modelo antes de decidir cómo desplegarlo.
2. Verificar qué hicieron el pruning y la quantization, y convertir a TFLite.
3. Servir un modelo con FastAPI (carga única, validación, `/health`).
4. Separar configuración y secretos con `.env`.
5. Empaquetar el servicio en Docker y comparar dos despliegues con docker compose.
6. Explicar el rol de SSH en el despliegue seguro.

## ✅ Qué se verificó
El notebook docente se ejecutó de principio a fin sin errores; la API (incluida la `API_KEY`) y `benchmark/compare.py` se probaron contra servicios reales. **Los `docker build` / `docker compose` y `ssh-keygen` no se pudieron probar en el entorno donde se preparó la clase**: haz una corrida de ensayo en tu máquina antes de clase.

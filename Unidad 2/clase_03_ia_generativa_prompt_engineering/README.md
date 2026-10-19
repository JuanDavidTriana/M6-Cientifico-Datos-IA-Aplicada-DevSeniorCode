# Unidad 2 · Clase 3: IA Generativa y Prompt Engineering — Proyecto **Devi** (parte 1)

**Módulo:** Científico de Datos e Inteligencia Artificial Aplicada · **Unidad 2:** NLP y el Auge de la IA Generativa (LLMs)

Las Clases 3 y 4 son un solo proyecto: **Devi, la asistente virtual de DevSenior Code**, construida por versiones.

| Versión | Clase | Cómo funciona |
|---|---|---|
| **Devi v1** | 3 (parte A) | GPT-2 en español ajustado (fine-tuning) con Alpaca + diálogos + la base de conocimiento de la empresa |
| **Devi v2** | 3 (parte B) | Qwen3-0.6B + system prompt con reglas + la información de la empresa **en el prompt** (buscador TF-IDF) |
| **Devi Pro** | 4 | Devi v2 + pipelines de Hugging Face (NER, sentimiento), BETO afinado para la intención y Qwen3 con LoRA |

## 🗂️ Contenido de la carpeta

| Archivo | Para quién | Descripción |
|---|---|---|
| `presentacion/Clase-03-Devi-IA-Generativa-Prompt-Engineering.pptx` | Docente | Diapositiva de apertura (con notas del orador) |
| `web/guia-clase-03-devi.html` | Docente | Guía interactiva para proyectar: agenda con reloj, demos y buscador TF-IDF real |
| `notebooks/Clase03_Devi_IA_Generativa_Prompt_Engineering_DOCENTE.ipynb` | Docente | Notebook **completo y funcional** con teoría extendida, guía de tiempos y preguntas frecuentes |
| `notebooks_estudiantes/Clase03_Devi_IA_Generativa_Prompt_Engineering_ESTUDIANTE.ipynb` | Estudiantes | Mismo notebook con las celdas de código **vacías** para live coding |
| `datasets/devsenior_conocimiento.jsonl` | Ambos | Base de conocimiento: 70 temas, 162 preguntas, con su **categoría** (intención) |
| `datasets/dialogos_minigpt.jsonl` | Ambos | 93 diálogos de la Clase 2 (personalidad y conceptos de IA) |
| `datasets/mensajes_devi.csv` | Ambos | 38 mensajes de interesados (con nombres, ciudades y reclamos) etiquetados con intención y sentimiento |
| `_anterior_minigpt/` | Docente | Versión anterior de la clase (MiniGPT + Qwen3), por si la necesitas. No se sube a GitHub |

## ⏱️ Agenda (3 horas)

| Tiempo | Sección del notebook |
|---|---|
| 0:00 – 0:10 | Intro: el proyecto Devi y cómo se construye un ChatGPT |
| 0:10 – 0:30 | A1–A2. Datos de Devi (3 fuentes) y formato de chat con máscara −100 |
| 0:30 – 0:55 | A3. Entrenar Devi v1 con presupuesto de tiempo (5–15 min en GPU) |
| 0:55 – 1:10 | A4. Hablar con Devi v1 y medir su memoria de la empresa |
| 1:10 – 1:20 | ☕ Pausa |
| 1:20 – 1:35 | B1. Qwen3 y chat templates |
| 1:35 – 1:45 | B2. Qwen3 sin datos de la empresa: la alucinación |
| 1:45 – 2:05 | B3. El system prompt de Devi |
| 2:05 – 2:25 | B4. Contexto en el prompt (buscador TF-IDF): **Devi v2** |
| 2:25 – 2:40 | B5. Zero-shot vs few-shot: intención del mensaje |
| 2:40 – 2:50 | B6. Salida JSON: la ficha del interesado |
| 2:50 – 2:58 | B7. Chat de Devi con memoria, contexto y streaming |
| 2:58 – 3:00 | Cierre y reto |

## ⚙️ Preparación (antes de la clase)

```bash
pip install -r requirements.txt
```
Ejecuta la sección 0, la celda A1.3 y la B1.1 con internet. Se descargan:
- `datificate/gpt2-small-spanish` (~500 MB) y una muestra de `bertin-project/alpaca-spanish`.
- `Qwen/Qwen3-0.6B` (~1,5 GB, Apache 2.0). Requiere `transformers >= 4.51`.

**Tiempos:** el entrenamiento de Devi v1 tiene tope de 15 min en GPU (5–10 min en una RTX 4070). En CPU el notebook usa datos reducidos y congela 6 capas (~10 min).
**Opcional:** si ya entrenaste la Devi de 2 h (`clase_02_transformers_minigpt/notebooks/Clase02_EXTRA_Devi_Asistente_DevSenior.ipynb`), la celda A4.3 la compara con la Devi de clase.

## 🎯 Objetivos
1. Explicar cómo se construye un asistente tipo ChatGPT (preentrenamiento → SFT → alineación).
2. Ajustar un modelo con datos de una empresa y medir qué memorizó.
3. Usar un LLM instruct con chat templates y escribir un buen system prompt.
4. Darle al modelo la información en el contexto (la idea de RAG) y compararlo con el fine-tuning.
5. Aplicar few-shot, salida JSON y construir un chat con memoria.

➡️ **Siguiente (Clase 4): Devi Pro** con pipelines, BETO afinado y LoRA.

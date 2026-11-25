# Unidad 2 · Clase 4 — Caso real: **Triage inteligente de PQRS con Transformers**

**Módulo:** Científico de Datos e Inteligencia Artificial Aplicada · **Unidad 2:** NLP y el Auge de la IA Generativa (LLMs)

Clase de recuperación, nivel avanzado, construida alrededor de **un problema real de punta a punta**. Una empresa de servicios públicos recibe ~3.000 PQRS al mes, tiene **15 días hábiles** para responder (Ley 142 de 1994, art. 158, con silencio administrativo positivo) y sus analistas pierden los casos graves entre cientos de mensajes.

Primero **abrimos el Transformer** para entender *qué analiza*: tokens, embeddings contextuales, self-attention y cómo llega a una decisión. Después lo usamos para construir la solución:

| Paso | Pregunta de negocio | Técnica |
|---|---|---|
| 1 | ¿Por qué no basta con reglas? | Baseline con palabras clave (y sus fallas) |
| 2 | ¿Cómo "lee" un Transformer? | Tokenizer, "corte" en 4 contextos capa por capa, mapa de atención, NLI por dentro |
| 3.1–3.2 | ¿Qué tipo es y qué área lo atiende? | Zero-shot NLI con etiquetas descriptivas |
| 3.3 | ¿Por qué lo clasificó así? | Explicabilidad por oclusión (palabra por palabra) |
| 3.4 / 3.7 | ¿Qué tan urgente es? | Sentimiento + señales de riesgo del negocio, recall de "alta" |
| 3.5 | ¿Quién, dónde, cuánto? | NER + regex y anonimización (Ley 1581 de 2012) |
| 3.6 | ¿Es un daño masivo? | Embeddings + clustering jerárquico |
| 3.8 | ¿Qué le respondemos? | Borrador con Qwen3 sobre datos anonimizados |
| 4 | ¿Qué atendemos primero? | Bandeja priorizada con fecha límite legal (días hábiles con festivos) |
| 5 | ¿Cómo nos apoyamos en la IA? | Impacto, humano en el loop y riesgos |

## 🗂️ Contenido

| Archivo | Para quién | Descripción |
|---|---|---|
| `notebooks/Clase04_Caso_Real_PQRS_Transformers_DOCENTE.ipynb` | Docente | Completo, con guía docente, discusión y preguntas frecuentes |
| `notebooks_estudiantes/Clase04_Caso_Real_PQRS_Transformers_ESTUDIANTE.ipynb` | Estudiantes | Celdas `✍️ LIVE CODING` vacías |
| `datasets/pqrs_servired.csv` | Ambos | 38 PQRS realistas etiquetadas (tipo, área, urgencia) con un daño masivo de energía (Kennedy) y uno de internet (Chapinero) |
| `generate_notebooks.py` | Docente | Regenera ambos notebooks |

## ⏱️ Agenda sugerida (≈ 2 h 30)

| Tiempo | Bloque |
|---|---|
| 0:00 – 0:10 | El problema y la arquitectura de la solución |
| 0:10 – 0:20 | 1. Por qué fallan las reglas |
| 0:20 – 1:00 | 2. Cómo "lee" un Transformer (el corazón conceptual) |
| 1:00 – 1:10 | ☕ Pausa |
| 1:10 – 2:05 | 3. La solución paso a paso |
| 2:05 – 2:30 | 4–5. Bandeja priorizada, impacto, riesgos y reto |

## ⚙️ Preparación
```bash
pip install torch "transformers>=4.51" sentence-transformers scikit-learn pandas matplotlib accelerate
```
Ejecuta las secciones 0 y 2 con internet antes de la clase. Modelos: BETO, zero-shot XNLI, RoBERTuito, NER, MiniLM multilingüe y Qwen3-0.6B (opcional: `GENERAR_RESPUESTAS = False`).

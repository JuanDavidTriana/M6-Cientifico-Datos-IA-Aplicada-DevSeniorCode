# Unidad 3 · Clase 1: Sistemas de Recomendación y Aprendizaje por Refuerzo

**Módulo:** Científico de Datos e Inteligencia Artificial Aplicada · **Unidad 3:** IA Aplicada, Databricks y Arquitecturas Modernas

**Caso:** la plataforma de cursos de DevSenior Code (40 cursos, 600 estudiantes, ~7 900 calificaciones).

| Parte | Pregunta de negocio | Técnicas |
|---|---|---|
| **A · Recomendación** | ¿Qué curso le mostramos a *cada* estudiante? | Popularidad bayesiana, contenido (TF-IDF), filtrado colaborativo item-item, SVD, factorización y NeuMF en TensorFlow, híbrido |
| **B · Refuerzo** | ¿Qué curso va en el banner? ¿En qué **orden** tomar los cursos para llegar al proyecto final? | Multi-armed bandits (ε-greedy, UCB1, Thompson), Q-learning, DQN (opcional) |

**Tecnologías:** Python, Scikit-Learn, TensorFlow/Keras, Pandas, Matplotlib.

## 🗂️ Contenido de la carpeta

| Archivo | Para quién | Descripción |
|---|---|---|
| `web/guia-clase-01-recomendacion-rl.html` | Docente | Guía interactiva para dar la clase: agenda con reloj, "dilo así" y analogías por sección, y demos vivas (TF-IDF, bandits, Q-learning) |
| `web/recorrido-notebook-clase-01.html` | Docente | Segunda pantalla: las 36 celdas del notebook una a una (qué hace, cómo explicarla, líneas clave, salida esperada, si falla). Se navega con ← → |
| `notebooks/Clase01_Recomendacion_Aprendizaje_Refuerzo_DOCENTE.ipynb` | Docente | Notebook **completo y ejecutado**, con guía de tiempos, notas por sección y preguntas frecuentes |
| `notebooks/Clase01_Recomendacion_Aprendizaje_Refuerzo_DOCENTE_GPU.ipynb` | Docente | Mismo notebook preparado para la **GPU NVIDIA** (`tensorflow[and-cuda]`), con demo de velocidad CPU vs GPU |
| `notebooks_estudiantes/Clase01_Recomendacion_Aprendizaje_Refuerzo_ESTUDIANTE.ipynb` | Estudiantes | Mismo notebook con las celdas de código **vacías** (y una pista) para live coding. Las celdas de apoyo vienen listas |
| `datasets/cursos.csv` | Ambos | 40 cursos: área, nivel (1–3), horas, etiquetas y descripción |
| `datasets/estudiantes.csv` | Ambos | 600 estudiantes: perfil (científico de datos, desarrollador web…), nivel inicial, ciudad, edad |
| `datasets/interacciones.csv` | Ambos | ~7 900 calificaciones (1–5), si completó el curso y la fecha |
| `datasets/generar_datos.py` | Docente | Regenera los tres CSV (datos sintéticos, semilla fija) |

> Los datos son **sintéticos**: cada perfil tiene gustos ocultos por área. El modelo nunca ve la columna `perfil`, y aun así los métodos colaborativos *redescubren* esos grupos (se ve en el gráfico de embeddings de A6).

## ⏱️ Agenda (3 horas)

| Tiempo | Sección del notebook |
|---|---|
| 0:00 – 0:10 | Intro: el caso y los tipos de recomendadores |
| 0:10 – 0:25 | A1. Datos, matriz dispersa, cola larga y split train/test |
| 0:25 – 0:35 | A2. Baseline de popularidad (promedio bayesiano) |
| 0:35 – 0:50 | A3. Basado en contenido (TF-IDF + coseno) |
| 0:50 – 1:10 | A4–A5. Filtrado colaborativo item-item y SVD |
| 1:10 – 1:20 | ☕ Pausa |
| 1:20 – 1:45 | A6. Factorización de matrices y NeuMF con TensorFlow |
| 1:45 – 2:00 | A7–A8. Comparación de métodos, arranque en frío e híbrido |
| 2:00 – 2:10 | B1. Teoría: agente, entorno, recompensa, exploración vs explotación |
| 2:10 – 2:30 | B2. Multi-armed bandits: el banner de la página de inicio |
| 2:30 – 2:55 | B3. Q-learning: la ruta de aprendizaje óptima |
| 2:55 – 3:00 | Cierre y reto (B4 · DQN queda como opcional/tarea) |

## ⚙️ Preparación

```bash
pip install -r requirements.txt
```

No necesita internet ni GPU: el notebook completo corre en **~1 minuto en CPU**.

## 🎯 Objetivos
1. Explicar los tipos de recomendadores: popularidad, contenido, colaborativo e híbrido.
2. Construir recomendadores con Scikit-Learn (TF-IDF, similitud coseno, SVD).
3. Entrenar una factorización de matrices y un Neural Collaborative Filtering con TensorFlow.
4. Evaluar con RMSE, precision@k, recall@k y cobertura; resolver el arranque en frío.
5. Entender agente, entorno, estado, acción y recompensa, y equilibrar exploración vs explotación con bandits.
6. Diseñar una estrategia de aprendizaje secuencial con Q-learning (y una DQN opcional).

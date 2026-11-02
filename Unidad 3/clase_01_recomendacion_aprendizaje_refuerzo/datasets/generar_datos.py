"""Genera los datasets sintéticos de la Clase 1 (Unidad 3).

Escenario: la plataforma de cursos de DevSenior Code. Cada estudiante tiene un perfil
(cientifico de datos, ingeniero de ML, ...) que define sus gustos ocultos por area.
Con esos gustos se simula que cursos toma y que calificación les da.

Uso:  python generar_datos.py      (escribe cursos.csv, estudiantes.csv, interacciones.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ESTUDIANTES = 600
CARPETA = Path(__file__).parent
rng = np.random.default_rng(SEED)

AREAS = ["Programación", "Datos", "Machine Learning", "Deep Learning", "NLP",
         "IA Generativa", "Web", "Cloud", "Habilidades"]

# id, titulo, area, nivel (1 básico - 3 avanzado), horas, etiquetas, descripcion
CURSOS = [
    ("C01", "Python desde cero", "Programación", 1, 30, "python logica variables funciones", "Aprende a programar en Python: variables, condicionales, ciclos y funciones con ejercicios prácticos."),
    ("C02", "Python intermedio y POO", "Programación", 2, 35, "python clases objetos modulos", "Programación orientada a objetos en Python, módulos, manejo de errores y buenas prácticas."),
    ("C03", "Git y GitHub para equipos", "Programación", 1, 12, "git github control versiones", "Control de versiones con Git, ramas, pull requests y trabajo colaborativo en GitHub."),
    ("C04", "Estructuras de datos y algoritmos", "Programación", 2, 40, "algoritmos complejidad python", "Listas, pilas, colas, árboles, grafos y análisis de complejidad para entrevistas técnicas."),
    ("C05", "Testing y código limpio", "Programación", 2, 20, "pytest testing calidad", "Pruebas unitarias con pytest, refactorización y principios de código limpio."),
    ("C06", "SQL para análisis de datos", "Datos", 1, 25, "sql consultas bases datos", "Consultas SQL, joins, agregaciones y funciones de ventana para analizar datos de negocio."),
    ("C07", "Pandas y NumPy", "Datos", 1, 28, "pandas numpy dataframes python datos", "Manipulación de datos con Pandas y NumPy: limpieza, transformaciones y agrupaciones."),
    ("C08", "Visualización de datos", "Datos", 1, 18, "matplotlib seaborn graficos datos", "Gráficos efectivos con Matplotlib y Seaborn para contar historias con datos."),
    ("C09", "Estadística para ciencia de datos", "Datos", 2, 30, "estadística probabilidad inferencia datos", "Probabilidad, distribuciones, pruebas de hipótesis e intervalos de confianza aplicados."),
    ("C10", "Power BI y tableros", "Datos", 1, 20, "powerbi dashboards negocio", "Tableros interactivos en Power BI para reportes de negocio y KPIs."),
    ("C11", "Ingeniería de datos con Spark", "Datos", 3, 35, "spark big data pipelines databricks", "Procesamiento distribuido con PySpark, pipelines de datos y lakehouse en Databricks."),
    ("C12", "Machine Learning con Scikit-Learn", "Machine Learning", 2, 40, "sklearn regresion clasificación machine learning", "Regresión, clasificación, validación cruzada y pipelines con Scikit-Learn."),
    ("C13", "Feature engineering", "Machine Learning", 2, 18, "variables preprocesamiento machine learning", "Creación y selección de variables, codificación de categorias y escalado."),
    ("C14", "Modelos de ensamble y XGBoost", "Machine Learning", 3, 22, "xgboost random forest boosting", "Random Forest, Gradient Boosting y XGBoost para competencias de Kaggle."),
    ("C15", "Aprendizaje no supervisado", "Machine Learning", 2, 20, "clustering kmeans pca", "Clustering con K-Means y DBSCAN, reducción de dimensionalidad con PCA."),
    ("C16", "Series de tiempo", "Machine Learning", 3, 24, "series tiempo pronóstico arima", "Pronóstico de demanda con ARIMA, Prophet y modelos de machine learning."),
    ("C17", "Sistemas de recomendación", "Machine Learning", 3, 20, "recomendación filtrado colaborativo embeddings", "Filtrado colaborativo, factorización de matrices y recomendadores neuronales."),
    ("C18", "MLOps y despliegue de modelos", "Machine Learning", 3, 30, "mlops mlflow docker despliegue", "Seguimiento de experimentos con MLflow, APIs con FastAPI y despliegue con Docker."),
    ("C19", "Redes neuronales con TensorFlow", "Deep Learning", 2, 32, "tensorflow keras redes neuronales deep learning", "Redes densas con Keras: capas, activaciones, optimizadores y regularizacion."),
    ("C20", "Visión artificial con CNN", "Deep Learning", 3, 30, "cnn imágenes visión deep learning", "Redes convolucionales, transfer learning y clasificación de imágenes."),
    ("C21", "Aprendizaje por refuerzo", "Deep Learning", 3, 25, "refuerzo q-learning agentes", "Agentes, recompensas, Q-learning y Deep Q-Networks con TensorFlow."),
    ("C22", "PyTorch práctico", "Deep Learning", 3, 28, "pytorch tensores deep learning", "Entrenamiento de redes en PyTorch: tensores, autograd y bucles de entrenamiento."),
    ("C23", "Text mining y embeddings", "NLP", 2, 22, "nlp texto embeddings tokenizacion", "Preprocesamiento de texto, TF-IDF, Word2Vec y embeddings para clasificar documentos."),
    ("C24", "Transformers y Hugging Face", "NLP", 3, 28, "transformers huggingface bert nlp", "Arquitectura Transformer, BERT y pipelines de Hugging Face para tareas de NLP."),
    ("C25", "Prompt engineering", "IA Generativa", 1, 12, "prompts llm chatgpt ia generativa", "Técnicas de prompting: zero-shot, few-shot, cadenas de razonamiento y salidas estructuradas."),
    ("C26", "Aplicaciones con LLMs y RAG", "IA Generativa", 3, 26, "llm rag embeddings agentes ia generativa", "Asistentes con LLMs, búsqueda semántica y RAG con bases de datos vectoriales."),
    ("C27", "Agentes de IA", "IA Generativa", 3, 20, "agentes herramientas llm automatizacion", "Agentes que usan herramientas, memoria y planificación para automatizar tareas."),
    ("C28", "HTML, CSS y JavaScript", "Web", 1, 35, "html css javascript frontend web", "Fundamentos del desarrollo web: maquetación, estilos y programación en el navegador."),
    ("C29", "React desde cero", "Web", 2, 30, "react frontend componentes web", "Componentes, estado, hooks y consumo de APIs con React."),
    ("C30", "APIs con FastAPI", "Web", 2, 18, "fastapi backend apis python web", "APIs REST con FastAPI, validación con Pydantic y documentación automatica."),
    ("C31", "Django full stack", "Web", 2, 36, "django backend python web", "Aplicaciones web completas con Django: modelos, vistas, plantillas y autenticación."),
    ("C32", "Docker y contenedores", "Cloud", 2, 15, "docker contenedores devops", "Imágenes, contenedores, docker-compose y buenas prácticas para desplegar aplicaciones."),
    ("C33", "AWS para científicos de datos", "Cloud", 2, 24, "aws s3 sagemaker nube", "Servicios de AWS para datos e IA: S3, Lambda y SageMaker."),
    ("C34", "Databricks y el lakehouse", "Cloud", 3, 20, "databricks spark lakehouse mlflow", "Notebooks colaborativos, Delta Lake y MLflow en la plataforma Databricks."),
    ("C35", "Arquitecturas cloud modernas", "Cloud", 3, 22, "arquitectura microservicios nube", "Microservicios, colas de mensajes, serverless y arquitecturas orientadas a eventos."),
    ("C36", "Comunicación de resultados", "Habilidades", 1, 8, "storytelling presentaciones", "Cómo presentar hallazgos de datos a audiencias no técnicas."),
    ("C37", "Inglés técnico para IT", "Habilidades", 1, 30, "inglés entrevistas documentación", "Vocabulario técnico, lectura de documentación y entrevistas en inglés."),
    ("C38", "Preparación para entrevistas tech", "Habilidades", 2, 10, "entrevistas carrera portafolio", "Portafolio, LinkedIn, pruebas técnicas y negociación salarial."),
    ("C39", "Ética e IA responsable", "Habilidades", 1, 8, "ética sesgos ia responsable", "Sesgos, privacidad, transparencia y regulación de sistemas de IA."),
    ("C40", "Metodologías ágiles y Scrum", "Habilidades", 1, 10, "scrum ágil equipos", "Trabajo en equipo con Scrum y Kanban en proyectos de software y datos."),
]

# Gustos de cada perfil por area (mismo orden que AREAS)
PERFILES = {
    "Científico de datos": [0.6, 1.0, 1.0, 0.6, 0.4, 0.3, 0.0, 0.2, 0.5],
    "Ingeniero de ML":     [0.7, 0.6, 1.0, 1.0, 0.6, 0.6, 0.1, 0.7, 0.2],
    "Analista de datos":   [0.3, 1.0, 0.5, 0.1, 0.1, 0.2, 0.0, 0.1, 0.8],
    "Desarrollador web":   [1.0, 0.2, 0.1, 0.0, 0.0, 0.3, 1.0, 0.7, 0.4],
    "Ingeniero de IA generativa": [0.7, 0.3, 0.5, 0.6, 1.0, 1.0, 0.3, 0.5, 0.3],
}
CIUDADES = ["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga", "Pereira", "Lima", "Ciudad de México", "Quito", "Santiago"]


def main():
    cursos = pd.DataFrame(CURSOS, columns=["curso_id", "titulo", "area", "nivel", "horas", "etiquetas", "descripcion"])
    calidad = rng.normal(0, 0.35, len(cursos))          # algunos cursos son simplemente mejores
    popularidad = rng.lognormal(0, 0.6, len(cursos))     # cola larga: pocos cursos muy vistos

    perfiles = rng.choice(list(PERFILES), N_ESTUDIANTES, p=[0.3, 0.2, 0.2, 0.2, 0.1])
    estudiantes = pd.DataFrame({
        "estudiante_id": [f"E{i:04d}" for i in range(1, N_ESTUDIANTES + 1)],
        "perfil": perfiles,
        "nivel_inicial": rng.choice([1, 2, 3], N_ESTUDIANTES, p=[0.5, 0.35, 0.15]),
        "ciudad": rng.choice(CIUDADES, N_ESTUDIANTES),
        "edad": rng.integers(18, 50, N_ESTUDIANTES),
    })

    area_idx = cursos["area"].map({a: i for i, a in enumerate(AREAS)}).to_numpy()
    filas = []
    for _, e in estudiantes.iterrows():
        gusto = np.clip(np.array(PERFILES[e.perfil]) + rng.normal(0, 0.25, len(AREAS)), 0, 1.3)
        afinidad = gusto[area_idx]
        ajuste_nivel = -0.5 * np.abs(cursos["nivel"].to_numpy() - e.nivel_inicial)
        prob = popularidad * np.exp(2.5 * afinidad)
        prob /= prob.sum()
        n = int(np.clip(rng.poisson(13), 4, 30))
        elegidos = rng.choice(len(cursos), n, replace=False, p=prob)
        for j in elegidos:
            z = 2.2 * afinidad[j] + ajuste_nivel[j] + calidad[j] + rng.normal(0, 0.55) - 0.9
            rating = int(np.clip(np.round(1 + 4 / (1 + np.exp(-z))), 1, 5))
            completado = rng.random() < 0.25 + 0.15 * rating
            fecha = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 640)))
            filas.append((e.estudiante_id, cursos.curso_id[j], rating, int(completado), fecha.date()))

    inter = pd.DataFrame(filas, columns=["estudiante_id", "curso_id", "rating", "completado", "fecha"])
    inter = inter.sort_values("fecha").reset_index(drop=True)

    cursos.to_csv(CARPETA / "cursos.csv", index=False)
    estudiantes.to_csv(CARPETA / "estudiantes.csv", index=False)
    inter.to_csv(CARPETA / "interacciones.csv", index=False)
    print(f"{len(cursos)} cursos · {len(estudiantes)} estudiantes · {len(inter)} interacciones")
    print(inter.rating.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

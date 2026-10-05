# Clase 01: Text Mining y Embeddings

Repositorio de apoyo para una clase de aproximadamente 4 horas del módulo **Científico de Datos e Inteligencia Artificial Aplicada**.

## Estructura del proyecto

```text
Clase-01-Text-Mining-y-Embeddings/
│
├── README.md
├── requirements.txt
├── datasets/
│   ├── frases.csv
│   ├── opiniones_clientes.csv
│   ├── noticias.csv
│   └── peliculas.csv
└── notebooks/
    ├── 01_Introduccion_Texto.ipynb
    ├── 02_Preprocesamiento.ipynb
    ├── 03_Tokenizacion.ipynb
    ├── 04_Padding.ipynb
    ├── 05_Representaciones_BoW_TFIDF.ipynb   ← nuevo
    ├── 06_Embedding_Layer.ipynb
    ├── 07_Mini_Clasificador_Sentimientos.ipynb
    └── 08_Laboratorio_Final.ipynb
```

## Instalación de dependencias

1. Crear y activar un entorno virtual (opcional, recomendado).
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Cómo ejecutar los notebooks

Desde la raíz del proyecto:

```bash
jupyter notebook
```

Abrir la carpeta `notebooks/` y ejecutar cada notebook en orden.

## Orden recomendado de estudio

1. `01_Introduccion_Texto.ipynb`
2. `02_Preprocesamiento.ipynb`
3. `03_Tokenizacion.ipynb`
4. `04_Padding.ipynb`
5. `05_Representaciones_BoW_TFIDF.ipynb`
6. `06_Embedding_Layer.ipynb`
7. `07_Mini_Clasificador_Sentimientos.ipynb`
8. `08_Laboratorio_Final.ipynb`

## Objetivos de aprendizaje por notebook

- **01 Introducción al Texto**: comprender por qué el texto debe transformarse antes de modelar.
- **02 Preprocesamiento**: limpieza paso a paso con regex y pandas, emojis, stopwords (riesgo de perder negaciones) y stemming.
- **03 Tokenización**: convertir texto a índices numéricos con `Tokenizer`.
- **04 Padding**: estandarizar longitudes de secuencias para redes neuronales.
- **05 Representaciones clásicas**: one-hot, Bag of Words, TF-IDF, similitud coseno y un mini buscador de noticias; sus limitaciones motivan los embeddings.
- **06 Embedding Layer**: la capa `Embedding` como tabla de búsqueda, shapes, parámetros y visualización PCA de lo que aprende.
- **07 Mini Clasificador de Sentimientos**: modelo end-to-end con split antes de tokenizar, `maxlen` por percentil, `mask_zero`, EarlyStopping, matriz de confusión y análisis de errores.
- **08 Laboratorio Final**: pipeline parametrizable, baseline TF-IDF + Regresión Logística y tabla de experimentos.

## Notas didácticas

- Cada notebook es independiente: importa sus propias librerías y carga datos por rutas relativas.
- Se muestran resultados intermedios con `print()` o `display()`.
- Todos los notebooks incluyen preguntas de reflexión para discusión en clase.
- Los datasets son pequeños y realistas para favorecer la comprensión conceptual.

## Datasets

| Archivo | Filas | Uso |
|---|---|---|
| `frases.csv` | 20 | Tokenización (NB 02–03) |
| `opiniones_clientes.csv` | 320 (160 pos / 160 neg, sin duplicados) | NB 01, 04, 06, 07 |
| `noticias.csv` | 12 | Buscador TF-IDF (NB 05) |
| `peliculas.csv` | 60 (30 / 30) | Laboratorio (NB 08) |

> **Cambio importante:** la versión anterior de `opiniones_clientes.csv` incluía la etiqueta dentro del texto (`"(caso positivo 1)"`) y repetía 10 frases 5 veces, lo que permitía al modelo "acertar" el 100 % por fuga de información. La nueva versión no tiene duplicados ni etiquetas en el texto.

## Compatibilidad

Probado con TensorFlow 2.21 / Keras 3. Se usa `keras.Input(shape=(maxlen,))` en lugar de `input_length` (obsoleto en Keras 3). `Tokenizer` y `pad_sequences` siguen disponibles como API legacy; el NB 03 muestra la alternativa moderna `TextVectorization`.

## Versión para estudiantes

La carpeta `notebooks_estudiantes/` tiene los mismos 8 notebooks con **toda la explicación** pero con las **celdas de código vacías**, para resolverlos en clase:

- La primera celda (imports y carga de datos) viene resuelta para arrancar rápido.
- Cada celda vacía indica el número de ejercicio, **pistas** (funciones útiles) y los **nombres de variables** que deben quedar definidos, porque las celdas siguientes los usan.
- La solución está en `notebooks/` (versión del docente, ya ejecutada).

Para el proyecto IMDb existe `IMDb_NLP_Processor/notebooks/actividad_estudiante.ipynb`.

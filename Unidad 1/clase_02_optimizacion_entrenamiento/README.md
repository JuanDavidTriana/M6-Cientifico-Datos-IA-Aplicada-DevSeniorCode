# Clase 2: Optimizacion y Entrenamiento de Redes Neuronales

## Caso de Estudio
Prediccion de supervivencia en el Titanic Dataset (Kaggle).

---

## Informacion General

- Modulo: Cientifico de Datos e Inteligencia Artificial Aplicada
- Unidad: 1
- Clase: 2
- Tema: Optimizacion y Entrenamiento de Redes Neuronales
- Tecnologias: Python, TensorFlow, Keras, Scikit-Learn, Matplotlib, Jupyter Notebook
- Dataset: Titanic - Machine Learning from Disaster (Kaggle)

---

## Objetivos de Aprendizaje

Al finalizar esta clase, el estudiante sera capaz de:

- Descargar y organizar datasets desde Kaggle (interfaz web y API oficial).
- Explicar como aprende una red neuronal mediante descenso de gradiente y backpropagation.
- Comparar optimizadores (SGD, Momentum, RMSprop, Adam) y elegir el adecuado.
- Ajustar el learning rate y aplicar schedulers de aprendizaje.
- Evaluar el efecto del batch size en la convergencia y la estabilidad.
- Aplicar inicializacion adecuada de pesos (Glorot, He).
- Regularizar redes con L1/L2, Dropout y Batch Normalization.
- Automatizar el entrenamiento con EarlyStopping, ModelCheckpoint y ReduceLROnPlateau.
- Diagnosticar overfitting y underfitting a partir de las curvas de entrenamiento.

---

## 0. Como Obtener Datasets de Kaggle

Kaggle es la fuente publica mas usada para datasets reales. En este modulo los datasets **no se suben al repositorio**: se documenta el enlace y cada estudiante los descarga en su entorno.

### Opcion A: descarga manual (recomendada para empezar)

1. Crear una cuenta gratuita en https://www.kaggle.com.
2. Entrar a la competencia o dataset. Para esta clase:
   - https://www.kaggle.com/competitions/titanic
3. Aceptar las reglas de la competencia (boton `Join Competition`) si es una competencia.
4. Ir a la pestana `Data` y descargar `train.csv` (y opcionalmente `test.csv`).
5. Guardar el archivo dentro de la carpeta de la clase:

```
clase_02_optimizacion_entrenamiento/
|-- data/
|   |-- train.csv
|-- U1_C2_optimizacion_entrenamiento_v1.ipynb
```

6. Verificar que `data/` este listado en el `.gitignore` del repositorio.

### Opcion B: API oficial de Kaggle (para automatizar)

1. Instalar la libreria:

```bash
pip install kaggle
```

2. Generar el token en https://www.kaggle.com/settings -> seccion `API` -> `Create New Token`. Se descarga un archivo `kaggle.json`.
3. Ubicar el token en la ruta que espera la libreria:

- Windows: `C:\Users\<usuario>\.kaggle\kaggle.json`
- Linux / macOS: `~/.kaggle/kaggle.json` (ejecutar `chmod 600 ~/.kaggle/kaggle.json`)

4. Descargar el dataset:

```bash
# Competencia (caso Titanic)
kaggle competitions download -c titanic -p data
# Dataset publico (formato usuario/nombre-dataset)
kaggle datasets download -d architsharma01/loan-approval-prediction-dataset -p data
```

5. Descomprimir:

```bash
# Windows PowerShell
Expand-Archive -Path data/titanic.zip -DestinationPath data
# Linux / macOS
unzip data/titanic.zip -d data
```

### Buenas practicas al usar datasets externos

- Nunca versionar el CSV ni el archivo `kaggle.json` en Git.
- Documentar siempre la fuente, la fecha de descarga y la licencia del dataset.
- No usar rutas absolutas del equipo personal; usar rutas relativas como `data/train.csv`.
- Revisar el diccionario de datos antes de modelar (pestana `Data` de Kaggle).

---

## 1. Por Que No Basta con Construir la Red

En la Clase 1 construimos una ANN que funcionaba. El problema es que **la misma arquitectura puede dar resultados muy distintos** segun como se entrene.

Dos modelos identicos pueden terminar en:

- Uno que no aprende (loss estancada).
- Uno que aprende y generaliza.
- Uno que memoriza el train y falla en test.

La diferencia no esta en las capas, esta en el **proceso de entrenamiento**: optimizador, learning rate, batch size, inicializacion, regularizacion y criterio de parada.

### Idea clave
Entrenar una red es resolver un problema de optimizacion. Optimizar bien es tan importante como disenar bien.

---

## 2. Nuestro Caso de Estudio: Titanic Dataset

Titanic es un dataset clasico de clasificacion binaria con datos reales e imperfectos (nulos, categoricas, outliers), ideal para practicar optimizacion.

### Variables de entrada principales

- `Pclass`: clase del pasajero (1, 2, 3)
- `Sex`: sexo
- `Age`: edad (contiene nulos)
- `SibSp`: hermanos / conyuges a bordo
- `Parch`: padres / hijos a bordo
- `Fare`: tarifa pagada
- `Embarked`: puerto de embarque (contiene nulos)

### Variable objetivo

- `Survived`: 0 (no sobrevivio) / 1 (sobrevivio)

### Objetivo del modelo
Predecir si un pasajero sobrevivio y, sobre ese problema, medir el impacto de cada decision de entrenamiento.

---

## 3. Como Aprende una Red: Descenso de Gradiente

El entrenamiento busca los pesos que minimizan la funcion de perdida.

Regla de actualizacion:

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$

Donde:

- $w$: peso
- $\eta$: learning rate (tasa de aprendizaje)
- $\frac{\partial L}{\partial w}$: gradiente de la perdida respecto al peso

### Backpropagation
Es el algoritmo que calcula esos gradientes propagando el error desde la salida hacia atras, capa por capa, usando la regla de la cadena.

### Ciclo de una epoca

1. Forward pass: la red predice.
2. Calculo de la perdida.
3. Backward pass: se calculan los gradientes.
4. Actualizacion de pesos con el optimizador.

### Variantes segun cuantos datos se usan por actualizacion

| Variante | Datos por paso | Caracteristica |
|---|---|---|
| Batch Gradient Descent | Todo el dataset | Estable pero lento |
| Stochastic (SGD) | 1 muestra | Rapido pero muy ruidoso |
| Mini-Batch | 32, 64, 128... | Equilibrio, es el estandar |

---

## 4. Optimizadores

Un optimizador define **como** se usa el gradiente para actualizar los pesos.

### SGD
Aplica la regla base. Simple y predecible, pero lento y sensible al learning rate.

### SGD con Momentum
Acumula la direccion de los pasos anteriores para atravesar zonas planas y reducir oscilaciones.

$$
v \leftarrow \beta v + (1-\beta)\frac{\partial L}{\partial w}, \quad w \leftarrow w - \eta v
$$

### RMSprop
Adapta el learning rate por parametro usando un promedio movil de los gradientes al cuadrado. Util en problemas con escalas muy distintas.

### Adam
Combina Momentum y RMSprop. Es el **default razonable** en la mayoria de casos.

### Tabla comparativa

| Optimizador | Ventaja | Cuando usarlo |
|---|---|---|
| SGD | Control total, buena generalizacion | Con scheduler y tiempo para ajustar |
| SGD + Momentum | Convergencia mas rapida y estable | Vision artificial, redes profundas |
| RMSprop | Learning rate adaptativo | Secuencias, RNN |
| Adam | Converge rapido sin mucho ajuste | Punto de partida por defecto |

---

## 5. Learning Rate: El Hiperparametro Mas Critico

| Learning rate | Sintoma en las curvas |
|---|---|
| Muy alto | Loss oscila, sube o se vuelve `NaN` |
| Muy bajo | Loss baja lentamente, el modelo no llega a converger |
| Adecuado | Loss baja rapido al inicio y se estabiliza |

### Schedulers (planificadores)
Reducen el learning rate durante el entrenamiento: pasos grandes al inicio para avanzar, pasos pequenos al final para afinar.

- `ExponentialDecay`: decae de forma continua.
- `PiecewiseConstantDecay`: escalones definidos manualmente.
- `ReduceLROnPlateau`: reduce el learning rate solo cuando la metrica de validacion deja de mejorar.

---

## 6. Batch Size

El batch size define cuantas muestras se procesan antes de actualizar los pesos.

| Batch size | Efecto |
|---|---|
| Pequeno (8-32) | Mas ruido, puede generalizar mejor, mas lento por epoca |
| Mediano (32-128) | Equilibrio habitual |
| Grande (256+) | Rapido y estable, puede caer en minimos que generalizan peor |

Regla practica: empezar en 32 y ajustar segun el tamano del dataset y la memoria disponible.

---

## 7. Inicializacion de Pesos

Si todos los pesos inician en cero, todas las neuronas aprenden lo mismo y la red no se diferencia. La inicializacion correcta mantiene la varianza de las senales entre capas.

| Inicializador | Recomendado con |
|---|---|
| Glorot / Xavier | `tanh`, `sigmoid` (default de Keras) |
| He | `relu` y variantes |

---

## 8. Regularizacion: Combatir el Overfitting

### L1 y L2 (weight decay)
Penalizan pesos grandes anadiendo un termino a la perdida.

$$
L_{total} = L + \lambda \sum w^2 \quad \text{(L2)}
$$

- L1 tiende a llevar pesos a cero (seleccion de variables).
- L2 reduce la magnitud de todos los pesos (mas usada).

### Dropout
Durante el entrenamiento desactiva aleatoriamente un porcentaje de neuronas. Obliga a la red a no depender de unas pocas rutas.

- Valores tipicos: 0.2 a 0.5.
- En inferencia Dropout se desactiva automaticamente.

### Batch Normalization
Normaliza las activaciones de cada mini-batch. Estabiliza el entrenamiento, permite learning rates mas altos y aporta un efecto regularizador leve.

---

## 9. Callbacks: Entrenamiento Automatizado

| Callback | Funcion |
|---|---|
| `EarlyStopping` | Detiene el entrenamiento cuando `val_loss` deja de mejorar |
| `ModelCheckpoint` | Guarda el mejor modelo segun la metrica de validacion |
| `ReduceLROnPlateau` | Reduce el learning rate cuando la mejora se estanca |

`EarlyStopping` con `restore_best_weights=True` es la forma correcta de elegir el numero de epocas: se entrena con un limite alto y el callback recupera el mejor punto.

---

## 10. Diagnostico: Leer las Curvas

| Situacion | Train | Validacion | Diagnostico | Accion |
|---|---|---|---|---|
| Ambas altas y cercanas | Baja loss | Baja loss | Buen ajuste | Consolidar |
| Train baja, val sube | Baja loss | Loss sube | Overfitting | Dropout, L2, EarlyStopping, mas datos |
| Ambas altas | Loss alta | Loss alta | Underfitting | Mas capas/neuronas, mas epocas, mayor lr |
| Loss oscila fuerte | Inestable | Inestable | Learning rate alto | Reducir lr, usar scheduler |
| Val mejor que train | - | - | Dropout activo en train | Normal, verificar en evaluacion |

---

## 11. Flujo de la Practica de Clase

1. Descargar el dataset Titanic desde Kaggle.
2. Cargar, limpiar e imputar valores nulos.
3. Preprocesar con `ColumnTransformer` (escalado + One-Hot Encoding).
4. Entrenar un modelo baseline como referencia.
5. Comparar optimizadores: SGD, Momentum, RMSprop, Adam.
6. Comparar learning rates y aplicar un scheduler.
7. Comparar batch sizes.
8. Comparar inicializadores de pesos.
9. Aplicar regularizacion: L2, Dropout, Batch Normalization.
10. Entrenar el modelo final con callbacks.
11. Evaluar con accuracy, reporte de clasificacion y matriz de confusion.
12. Concluir que decisiones aportaron mejora real.

---

## 12. Resultado Esperado

Al finalizar, el estudiante deberia poder:

- Justificar con evidencia (curvas y metricas) cada decision de entrenamiento.
- Pasar de un baseline a un modelo optimizado de forma reproducible.
- Detectar overfitting y aplicar la correccion adecuada.
- Documentar un experimento de forma comparable.

---

## Recursos Practicos de la Clase

- `U1_C2_optimizacion_entrenamiento_v1.ipynb`: notebook resuelto de la sesion.
- `U1_C2_optimizacion_entrenamiento_v1 clase.ipynb`: version para codificar en vivo.

### Dataset
- Titanic - Machine Learning from Disaster: https://www.kaggle.com/competitions/titanic
- Archivo requerido: `train.csv` ubicado en `data/` dentro de la carpeta de la clase.

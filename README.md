# 🎬 Análisis de Sentimiento en Reseñas de Películas (IMDB)

Este repositorio contiene un proyecto práctico de **Procesamiento de Lenguaje Natural (PLN)** enfocado en la clasificación binaria de opiniones. El objetivo principal es predecir si una reseña cinematográfica es **positiva** o **negativa** a partir del texto escrito por el usuario.

El modelo está construido sobre el popular conjunto de datos **IMDB Dataset**, utilizando técnicas clásicas de vectorización de texto y algoritmos de aprendizaje automático (Machine Learning) supervisado.

---

## 🚀 Características del Proyecto

*   **Balanceo de Clases:** Selección de un subconjunto equitativo de 10,000 reseñas (5,000 positivas y 5,000 negativas) para garantizar un entrenamiento balanceado.
*   **Procesamiento de Texto (TF-IDF):** Extracción de características numéricas mediante *Term Frequency - Inverse Document Frequency* (`TfidfVectorizer`), eliminando palabras vacías (*stop words*) en inglés.
*   **Clasificador de Soporte Vectorial (SVM):** Uso de una máquina de vectores de soporte (`SVC`) con núcleo lineal para realizar la clasificación, un algoritmo altamente eficiente para datos de texto de alta dimensionalidad.
*   **Evaluación Completa:** Cálculo de métricas estándar en ciencia de datos como precisión (*Accuracy*), puntuación F1 (*F1-Score*), reporte detallado de clasificación y matriz de confusión.

---

## 📁 Estructura del Repositorio

```text
procesamiento_ln/
├── data/
│   └── IMDB Dataset.csv       # Conjunto de datos con 50,000 reseñas de IMDB
├── 1_analisis_sentimiento.ipynb # Notebook interactivo de Jupyter con todo el flujo
├── requirements.txt           # Dependencias y paquetes de Python necesarios
├── sentiment.png              # Visualización de los resultados del análisis
├── LICENSE                    # Licencia MIT de distribución
└── README.md                  # Documentación del proyecto (este archivo)
```

---

## 🛠️ Requisitos e Instalación

Para ejecutar este proyecto de forma local, sigue estos pasos:

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/procesamiento_ln.git
cd procesamiento_ln
```

### 2. Configurar un entorno virtual (Recomendado)
Crear un entorno virtual con `venv` te ayuda a mantener limpias tus dependencias:

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar las dependencias
Instala todas las librerías necesarias con el gestor de paquetes de Python (`pip`):
```bash
pip install -r requirements.txt
```

Las dependencias principales instaladas son:
*   `pandas` y `numpy` para manipulación de datos.
*   `scikit-learn` para el procesamiento de texto y modelado predictivo.
*   `matplotlib` y `seaborn` para la visualización de datos.
*   `jupyterlab` para ejecutar el cuaderno interactivo.

---

## 💻 Flujo de Trabajo en el Notebook

El cuaderno [`1_analisis_sentimiento.ipynb`](./1_analisis_sentimiento.ipynb) está estructurado de la siguiente manera:

1.  **Carga de Datos:** Lectura del archivo `IMDB Dataset.csv` desde la carpeta `data/`.
2.  **Exploración y Filtrado:** Inspección inicial de los datos y extracción de un subconjunto balanceado de 10,000 reseñas para agilizar el procesamiento sin sacrificar la representatividad.
3.  **División de Datos:** Separación de los datos en conjuntos de **entrenamiento (67%)** y **prueba (33%)** con `train_test_split`.
4.  **Extracción de Características:** Transformación del texto a formato numérico con `TfidfVectorizer(stop_words='english')`.
5.  **Entrenamiento del Modelo:** Ajuste de un clasificador lineal `SVC(kernel='linear')`.
6.  **Pruebas de Inferencia:** Evaluación manual del modelo ingresando frases personalizadas (ej. *"A good movie"*, *"I did not like this movie at all"*).
7.  **Evaluación de Métricas:**
    *   **Precisión (*Accuracy*):** Porcentaje total de predicciones correctas.
    *   **Puntuación F1 (*F1-Score*):** Media armónica de precisión y exhaustividad para ambas clases (`positive` y `negative`).
    *   **Reporte de Clasificación:** Tabla detallada con *precision*, *recall* y *f1-score*.
    *   **Matriz de Confusión:** Tabla que muestra la distribución de verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.

---

## 📊 Ejecución del Proyecto

Una vez que las dependencias estén instaladas y tu entorno activado, inicia Jupyter Lab:

```bash
jupyter lab
```

Se abrirá una pestaña en tu navegador web. Abre el archivo `1_analisis_sentimiento.ipynb` y ejecuta las celdas secuencialmente para ver el proceso y replicar los resultados del entrenamiento.

También se incluye una versión reproducible desde la terminal, con limpieza de HTML,
división estratificada, TF-IDF, SVM lineal y generación automática de métricas:

```bash
python sentiment_analysis.py
```

Los resultados se guardan en `results/metrics.json` y
`results/confusion_matrix.png`. Para entrenar con las 50,000 reseñas:

```bash
python sentiment_analysis.py --samples-per-class 25000
```

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](./LICENSE) para obtener más detalles.

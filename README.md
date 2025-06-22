# Detector de Sentimientos con Machine Learning

Este proyecto implementa un sistema de análisis de sentimientos en español que integra:

- **Generación de dataset diverso:**  
  Incluye comentarios formales, informales, con errores de tipeo y ruido.
- **Preprocesamiento refinado:**  
  Con manejo de negación y preservación de signos de puntuación importantes.
- **Modelos optimizados:**  
  Pipelines con Regresión Logística y Multinomial Naive Bayes, con GridSearchCV y validación cruzada.
- **Interfaz interactiva:**  
  Una aplicación web (Streamlit) para analizar nuevos comentarios en tiempo real y visualizar la distribución del dataset.

## Estructura del Proyecto

ns_ds/ ├── data/ │ └── dataset.csv ├── src/ │ ├── init.py │ ├── preprocess.py │ ├── sentiment_analysis.py │ └── model_interface.py ├── generate_dataset.py ├── app.py ├── README.md └── .gitignore


## Instrucciones

1. **Crear el entorno virtual:**
   ```bash
   python -m venv venv
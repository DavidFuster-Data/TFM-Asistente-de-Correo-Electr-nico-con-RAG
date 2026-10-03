# Asistente de Correo Electrónico (RAG) — TFM

Trabajo de Fin de Máster en Big Data y Análisis de Datos (VIU): un asistente de correo electrónico basado en *Retrieval-Augmented Generation* (RAG), aplicado al dataset público de Enron.

## Contenido del repositorio

- **TFM**: Notebook principal de la investigación, que abarca todas las fases y procesos del trabajo.
- **TFM_A2**: En este notebook se aplicó la segunda técnica A.2, `SemanticChunker` (`langchain_experimental`), al dataset de Sally Beck, y se realizó la indexación y el análisis. Se hizo aparte para no saturar el notebook principal.
- **TFM_Mark_Taylor**: En este notebook se crea e indexa el dataset de Mark Taylor, adaptando al nuevo empleado la misma lógica que se usó con Sally Beck. Queda fuera del notebook principal debido a que se realizó en fases finales del trabajo, al requerir de dos personas para la aplicación.
- - **CSV_TFM**: Carpeta de archivos con los datos y resultados más relevantes del TFM, en formato CSV.
- **App**: Código de la aplicación en Streamlit (`app.py`). Se comparte únicamente el código fuente; quedan fuera del repositorio las bases de datos vectoriales ya indexadas y la clave de la API de Gemini, por lo que la aplicación no es ejecutable directamente al clonar el repositorio.

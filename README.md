# 🔍 Explorador de Sesgos: Auditoría Algorítmica en el Aula

> **Herramienta educativa para enseñar Machine Learning con un enfoque Ético y Pensamiento Crítico.**

## 📖 Introducción
Este proyecto es una mini aplicación interactiva diseñada para formadores y estudiantes dentro del ámbito de la Ciencia de Datos. Cuyo propósito es demostrar cómo los algoritmos de Machine Learning, aunque sean matemáticamente correctos, pueden heredar y amplificar los sesgos sociales existentes en los datos históricos.

EL dataset utilizado para este ejemplo es: **Adult Census Income** para entrenar modelos que predicen si una persona gana más de $50k/año, revelando disparidades de género en las predicciones.
URL del dataset: https://archive.ics.uci.edu/dataset/2/adult

---

## ⚙️ 1. ¿Qué hace el código? (Paso a Paso)

El proyecto sigue una arquitectura modular separando la lógica (`src`) de la visualización (`app.py`).

1.  **Ingesta de Datos (`src/logic.py`):**
    * Descarga el dataset desde el repositorio UCI.
    * Convierte la variable objetivo a binaria (0: <=50K, 1: >50K).
    * Limpia valores nulos y espacios en blanco (Eliminando el ruido del dataset) .

2.  **Preprocesamiento y Pipeline:**
    * Divide los datos en entrenamiento y prueba (`train_test_split`) con una semilla fija (`42`) para que todos los usuarios, en este caso los estudiantes obtengan el mismo resultado.
    * Aplica **OneHotEncoding** a variables categóricas (como 'job', 'marital-status') y **StandardScaler** a numéricas.

3.  **Entrenamiento:**
    * Entrena dos modelos contrastantes: **Regresión Logística** (lineal/interpretable) y **Random Forest** (no lineal/complejo).

4.  **Evaluación y Auditoría:**
    * Calcula métricas estándar (Accuracy, F1).
    * **Paso Crítico:** Desglosa el *Recall* por género para medir la equidad.
    * Genera curvas ROC y Matrices de Confusión.

5.  **Interfaz (`app.py`):**
    * Visualiza todo lo anterior usando **Streamlit**.
    * Simula una "IA GEnerativa y Explicable" (Mock) que traduce los resultados técnicos a lenguaje natural.

---

## 🎓 2. Objetivos de Aprendizaje

Al completar esta actividad, se espera aprender lo siguiente:

* **Técnicos:** Implementar un flujo completo de ML (limpieza -> entreno -> métricas) usando *Scikit-Learn* y *Pipelines*.
* **Analíticos:** Interpretar una **Matriz de Confusión** y entender por qué el *Accuracy* es una métrica engañosa en datasets que estén parcialmente desbalanceados.
* **Éticos:** Identificar un **Sesgo Algorítmico** cuantificable (diferencia de Recall entre hombres y mujeres) y comprender el impacto social de los Falsos Negativos.
* **Críticos:** Cuestionar la "objetividad" de la tecnología y la importancia de la *calidad*,antes que cantidad de los datos.

---

## 🏫 3. Guía Didáctica para Clase (90 - 120 Minutos)

Esta herramienta está pensada para una sesión de taller guiado para trabajarlo en clase.

### 🕒 Fase 1: Configuración y Contexto (15 - 30 min)
* **Actividad:** Clonar repo, instalar `requirements.txt` y lanzar la app.
* **Discusión:** *"¿Creen que una IA puede tener un sesgo al punto de llegar a ser machista? ¿Por qué?"*
* **Exploración:** Mirar el dataset en la App. Identificar columnas sensibles (Raza, Sexo, País) y cómo pueden llegar a influir a la hora de la toma de decisiones.

### 🕒 Fase 2: La Trampa de la Eficiencia (20 - 40 min)
* **Actividad:** Entrenar el modelo **Random Forest**.
* **Observación:** Ver que el *Accuracy* es alto (~85%).
* **Pregunta Trampa:** *"El modelo acierta el 85% de las veces. ¿Se pondría lanzar en un entorno de producción en un banco mañana mismo?"* (La mayoría probablemente dirá que sí).

### 🕒 Fase 3: Auditoría Forense (30 - 60 min)
* **Actividad:** Navegar a las pestañas de **Matriz de Confusión** y **Análisis de Sesgos** y explicar conceptos.
* **El "Eureka":** Descubrir que el modelo predice muy bien la riqueza en hombres, pero falla mucho más en mujeres (Recall bajo).
* **Concepto:** Explicar los **Falsos Negativos**. *En este contexto, un Falso Negativo es una mujer solvente a la que se le niega el crédito injustamente.*

### 🕒 Fase 4: IA Generativa y Cierre (25 - 45 min)
* **Actividad:** Leer la explicación del Mock de IA.
* **Debate Final:** ¿Cómo se podría arreglar esto?
    * *Idea 1:* ¿Borrar la columna "Sexo"? (Discutir variables proxy).
    * *Idea 2:* Conseguir más datos de mujeres ricas (Representatividad).
    * *Idea 3:* Algoritmos de "Fairness" (Discriminación positiva matemática).

---

## ⚠️ 4. Qué puede salir mal y cómo solucionarlo

### 🔴 Error: `ModuleNotFoundError: No module named 'src'`
* **Causa:** Python no encuentra la carpeta de lógica.
* **Solución:** Asegurarse de ejecutar `streamlit run app.py` desde la raíz del proyecto (donde está el README), no desde dentro de carpetas. Verifica que `src` tiene un archivo `__init__.py` vacío.

### 🔴 Error: `NameError: name 'null' is not defined`
* **Causa:** Se copió código crudo de un Jupyter Notebook (JSON) a un archivo `.py`.
* **Solución:** Limpiar el archivo `src/__init__.py` (debe estar vacío) y revisar que `src/logic.py` sea código Python puro.

### 🔴 Error de Datos: Fallo en la descarga del CSV
* **Causa:** La URL del repositorio UCI a veces se cae o cambia.
* **Solución:** El código está diseñado para buscar primero en internet. Si falla, el usuario debe descargar el archivo `adult.data` y colocarlo manualmente en una carpeta `data/` local.

### 🔴 API Keys de OpenAI
* **Nota:** Esta mini app usa un **MOCK** (simulación) para la parte de IA Generativa.
* **Ventaja:** No necesitas API Keys, no hay costes y nunca fallará la conexión en medio de la clase. Es reproducible 100%.

---

## 🚀 5. Adaptación por Niveles

### 🐣 Nivel Principiante (Sin Código)
* **Enfoque:** Usar solo la interfaz gráfica.
* **Actividad:** Comparar visualmente Regresión Logística vs. Random Forest. Centrarse en la interpretación ética de los gráficos sin tocar el Python.

### 🦁 Nivel Intermedio (Bootcamp/Junior)
* **Enfoque:** Tocar el código en `src/logic.py`.
* **Reto:** Cambiar los hiperparámetros del Random Forest (ej. `max_depth`) y ver cómo afecta al sesgo. ¿Un modelo más inteligente es más justo?

### 🐲 Nivel Avanzado (Senior/Máster)
* **Enfoque:** Implementar mitigación de sesgos.
* **Reto:** Modificar el pipeline para incluir `SMOTE` (balanceo de clases) o eliminar la columna `sex` antes de entrenar y medir si el sesgo desaparece o persiste por correlaciones ocultas.

---
*Desarrollado como proyecto personal y prueba Técnica para Puesto Formadora IA. DEC2025*

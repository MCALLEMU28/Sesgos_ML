import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

try:
    from src.logic import BiasExplorerModel
except ImportError:
    
    from src.bias_model import BiasExplorer as BiasExplorerModel

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Explorador de Sesgos", layout="wide", page_icon="⚖️")

# --- TÍTULO Y CONTEXTO ---
st.title("⚖️ Explorador de Sesgos en Algoritmos")
st.markdown("""
**Simulador Pedagógico:** Entrena una Inteligencia Artificial y audita sus decisiones. 
Descubre cómo un modelo puede tener buenas matemáticas pero mala ética.
""")

# --- GESTIÓN DE ESTADO (SINGLETON) ---
# Esto evita que el modelo se borre cada vez que el usuario toca un botón
if 'explorer' not in st.session_state:
    st.session_state.explorer = BiasExplorerModel()

explorer = st.session_state.explorer

# ==========================================
# 1. CARGA, DESCRIPCIÓN Y LIMPIEZA
# ==========================================
st.header("1. Datos y Limpieza")

# Pestañas para separar la visualización de la descripción técnica
tab_data, tab_info = st.tabs(["📂 Carga y Exploración", "ℹ️ Descripción del Dataset"])

with tab_data:
    col_load, col_clean = st.columns(2)
    
    with col_load:
        if st.button("📥 Cargar Datos Crudos"):
            with st.spinner("Descargando desde UCI Repository..."):
                explorer.load_data()
                st.success(f"Cargados {explorer.data.shape[0]} registros.")
    
    with col_clean:
        if explorer.data is not None:
            if st.button("🧹 Limpiar Outliers (IQR)"):
                removed = explorer.clean_outliers()
                st.success(f"Se eliminaron {removed} registros atípicos (Outliers).")
                st.caption("Filas restantes: " + str(explorer.data.shape[0]))

    if explorer.data is not None:
        st.dataframe(explorer.data.head(), use_container_width=True)
        st.markdown(f"**Target:** Columna `target` (1 = >50K, 0 = <=50K)")

with tab_info:
    st.markdown("""
    ### 📝 Ficha Técnica del Dataset
    
    * **Nombre:** Adult Census Income Dataset.
    * **Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult).
    * **Licencia:** CC BY 4.0 (Creative Commons Attribution).
    * **Tamaño Original:** ~32,561 filas (Train) y 15 columnas.
    * **Target (Objetivo):** Predecir si el ingreso anual supera los $50,000.
    
    ### 🧼 Proceso de Limpieza
    1.  **Nulos:** Se imputan con la Mediana (numéricos) o la Moda (categóricos).
    2.  **Duplicados:** Se eliminan filas idénticas.
    3.  **Outliers:** Aplicamos **Rango Intercuartílico (IQR)** para eliminar edades o horas de trabajo extremas e inverosímiles.
    4.  **Codificación:** Las variables de texto (ej. 'Job') se convierten a números con *OneHotEncoding*.
    """)

# ==========================================
# 2. ENTRENAMIENTO
# ==========================================
if explorer.data is not None:
    st.header("2. Entrenamiento (El Aprendizaje)")
    
    col_conf, col_btn = st.columns([3, 1])
    
    with col_conf:
        st.markdown("**Modelos a comparar:**")
        st.markdown("* 🧠 **Regresión Logística:** Simple, lineal, fácil de explicar.")
        st.markdown("* 🌲 **Random Forest:** Complejo, robusto, pero una 'caja negra'.")
        
    with col_btn:
        if st.button("⚙️ Entrenar Modelos"):
            with st.spinner("Entrenando inteligencias..."):
                # Se asume que preprocess_and_split o preprocess_data existe
                # Se intenta llamar al método correcto según la versión del script logic.py
                try:
                    dims = explorer.preprocess_and_split(test_size=0.2)
                except AttributeError:
                    dims = explorer.preprocess_data(test_size=0.2)
                
                explorer.train_models()
                st.success(f"¡Modelos listos! (Test set: {dims[1]} muestras)")

# ==========================================
# 3. AUDITORÍA TÉCNICA Y VISUALIZACIÓN
# ==========================================
if explorer.models:
    st.divider()
    st.header("3. Auditoría del Modelo (El Examen)")
    
    # Selección de modelo
    selected_model = st.selectbox("🤖 ¿Qué modelo se quiere auditar?", list(explorer.models.keys()))
    
    # Obtener métricas y gráficos
    # NOTA: Asegurar de que la función evaluate_model en logic.py devuelva 4 valores
    metrics, cm, roc_data, y_pred = explorer.evaluate_model(selected_model)
    
    # --- PESTAÑAS VISUALES ---
    tab1, tab2, tab3 = st.tabs(["📊 Métricas Clave", "🔲 Matriz de Confusión", "📈 Curva ROC"])
    
    # PESTAÑA 1: KPIs
    with tab1:
        st.subheader("Rendimiento General")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Accuracy", f"{metrics['Accuracy']:.2%}", help="% de aciertos totales.")
        kpi2.metric("Recall (Sensibilidad)", f"{metrics['Recall']:.2%}", help="De los que ganan >50K, ¿cuántos se lograron detectar?")
        kpi3.metric("F1 Score (Weighted)", f"{metrics['F1 Weighted']:.2f}", help="Métrica balanceada para datos desequilibrados.")
        
        st.info("""
        **💡 Interpretación Pedagógica:**
        * Si el **Accuracy** es alto pero el **Recall** es bajo, el modelo es "perezoso": predice que todos son pobres (clase mayoritaria) y acierta por estadística, pero falla en encontrar los casos relevantes.
        * El **F1 Weighted** es la métrica más honesta aquí porque los datos están desbalanceados.
        """)

    # PESTAÑA 2: MATRIZ DE CONFUSIÓN
    with tab2:
        col_graph, col_txt = st.columns([2, 1])
        with col_graph:
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                        xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
            ax_cm.set_ylabel('Realidad (Lo que es)')
            ax_cm.set_xlabel('Predicción (Lo que dice la IA)')
            st.pyplot(fig_cm)
        
        with col_txt:
            st.markdown("### ¿Cómo leer esto?")
            st.write("**Diagonal oscura:** Predicciones correctas.")
            st.error(f"**Falsos Negativos (Abajo-Izq):** {cm[1][0]} personas.")
            st.caption("▲ Estas son personas que ganan >50K pero la IA dijo que NO. En un banco, sería gente solvente a la que negamos un crédito injustamente.")

    # PESTAÑA 3: CURVA ROC
    with tab3:
        fpr, tpr, roc_auc = roc_data
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlabel('Tasa de Falsos Positivos (Ruido)')
        ax_roc.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        st.markdown("**Interpretación:** Cuanto más se pegue la curva naranja a la esquina superior izquierda, mejor es el modelo discriminando entre clases.")

# ==========================================
# 4. ANÁLISIS DE SESGOS Y ÉTICA
# ==========================================
if explorer.models:
    st.divider()
    st.header("4. La Mirada Ética")
    
    col_bias_chart, col_bias_text = st.columns([2, 1])
    
    # Si get_bias_metrics o analyze_bias existe, úsalo
    try:
        bias_data = explorer.get_bias_metrics(selected_model, sensitive_column='sex')
    except AttributeError:
        bias_data = explorer.analyze_bias(y_pred, sensitive_column='sex')

    with col_bias_chart:
        fig_bias, ax_bias = plt.subplots(figsize=(6, 4))
        sns.barplot(x=list(bias_data.keys()), y=list(bias_data.values()), palette="viridis", ax=ax_bias)
        ax_bias.set_title("Recall por Género (Capacidad de detectar riqueza)")
        ax_bias.set_ylim(0, 1)
        st.pyplot(fig_bias)

    with col_bias_text:
        st.warning("⚠️ **Alerta de Sesgo Detectada**")
        
        # Se recupera los valores exactos (asegurando claves sin espacios extra)
        # Nota: El dataset a veces usa ' Male' y otras 'Male', el .strip() ayuda a asegurar
        recall_male = bias_data.get('Male', bias_data.get(' Male', 0))
        recall_female = bias_data.get('Female', bias_data.get(' Female', 0))
        
        # Mostramos métricas grandes
        st.metric("Recall Hombres", f"{recall_male:.2%}")
        st.metric("Recall Mujeres", f"{recall_female:.2%}", delta=f"-{(recall_male - recall_female):.2%}")
        
        st.markdown(f"""
        **Interpretación:**
        Existe una brecha del **{abs(recall_male - recall_female):.2%}** en el rendimiento.
        
        Esto indica que el modelo tiene muchas más probabilidades de **ignorar el éxito financiero** si el perfil pertenece a una mujer, perpetuando la desigualdad histórica.
        """)

    # ==========================================
    # 5. IA GENERATIVA (MOCK)
    # ==========================================
    st.subheader("🤖 Explicación Generativa (Simulación LLM)")
    
    with st.expander("Ver Prompt enviado al LLM"):
        prompt = f"""
        Actúa como experto en ética. Analiza:
        Modelo: {selected_model}
        Recall Global: {metrics['Recall']:.2f}
        Recall por género: {bias_data}
        Explica a un estudiante por qué esto es injusto.
        """
        st.code(prompt)
    
    st.markdown(f"""
    > **Respuesta de la IA:**
    >
    > "Se ha analizado el modelo **{selected_model}**. Aunque tiene un Accuracy decente, se detecta un comportamiento discriminatorio.
    >
    > Fíjate en el Recall de las mujeres ({bias_data.get(' Female', 0):.2f}) comparado con el de los hombres ({bias_data.get(' Male', 0):.2f}). 
    >
    > **¿Qué significa esto en la vida real?**
    > El algoritmo está penalizando a las mujeres, fallando más al reconocer sus ingresos. Esto no es culpa del algoritmo matemático, sino de los datos del año 1994 que usó para entrenarlo. **Has digitalizado un prejuicio del pasado.**"
    """)

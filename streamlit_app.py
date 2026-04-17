import streamlit as st
import pandas as pd
from pathlib import Path

from scripts.conversor_json_csv import convertir_json_to_df
from scripts.preprocesamiento import preprocesar_dataframe
from scripts.analisis import analizar_datos
from modelos.regresion_lineal import analizar_regresion
from modelos.agrupamiento import analizar_agrupamiento
from modelos.regresion_logistica import analizar_regresion_logistica
from modelos.regresion_logistica import predecir_hora_desaparicion

# Paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
MAIN_CSV = DATA_DIR / 'desaparecidos_mx.csv'

st.set_page_config(page_title="Desaparecidos en México (Streamlit)", layout="wide", initial_sidebar_state="expanded")

# ----------- SIDEBAR -----------
st.sidebar.header("Aprendizaje Incremental y Datos para Regresion Logistica")
st.sidebar.markdown("Sube los registros para comenzar el procesamiento de los mismos, mostrando analisis basico, exploracion visual, modelos de regresion lineal, regresion logistica y agrupamiento,  en caso de subir mas de un archivo, se combinaran en uno solo, asi como tambien se almacenara para el entrenamiento del modelo logistico, que busca predecir la hora en la que una posible desaparicion pudo ocurrir, en base a la edad, sexo, dia y mes del hecho.")
uploaded_file = st.sidebar.file_uploader("Sube un nuevo Dataset", type=["json", "csv", "xlsx", "xls"])

df_master = pd.DataFrame()

if uploaded_file is not None:
    with st.spinner('Procesando nuevos datos e integrando incrementalmente...'):
        if uploaded_file.name.endswith('.json'):
            df_crudo = convertir_json_to_df(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df_crudo = pd.read_csv(uploaded_file)
        else:
            df_crudo = pd.read_excel(uploaded_file)
            
        df_nuevo_limpio = pd.DataFrame()
        df_nuevo_limpio = preprocesar_dataframe(df_crudo)
        
        if not df_nuevo_limpio.empty:
            df_nuevo_limpio.to_csv(MAIN_CSV, index=False)
            
            # Solo Regresión Logística mantendrá aprendizaje incremental real.
            analizar_regresion_logistica(df_nuevo_limpio, is_incremental=True)
            
            df_master = df_nuevo_limpio
            st.cache_data.clear()
            st.sidebar.success(f"¡Base de datos combinada a {len(df_nuevo_limpio)} nuevos casos y Modelo ROC Logístico iterado!")
        else:
            st.sidebar.error("El archivo cargado no contenía datos válidos o no se pudo procesar.")

st.sidebar.subheader("Predecir parte del dia de una desaparicion")
st.sidebar.markdown("Dado un perfil de persona desaparecida, carga el modelo logístico guardado y predice si la desaparición ocurrió de Día o de Noche, junto con su probabilidad.")
sexo = st.sidebar.radio(
    "Genero de la persona",
    [0, 1],
    captions=[
        "Mujer",
        "Hombre",
    ],
)
edad = st.sidebar.number_input(
    "Ingresa la Edad", value=None, placeholder="Edad de la persona cuando se vio por ultima vez"
)
dia_semana = st.sidebar.number_input(
    "Ingresa el Dia de la Semana", value=None, placeholder="0 = Lunes ... 6 = Domingo", max_value = 6
)
mes = st.sidebar.number_input(
    "Ingresa el Mes", value=None, placeholder="1 = Enero ... 12 = Diciembre", max_value = 12
)

if st.sidebar.button("Predecir..."):
    prediccion = predecir_hora_desaparicion(sexo, edad, dia_semana, mes)
    if prediccion:
        st.sidebar.write(f"Predicción: {prediccion['etiqueta']}")
        st.sidebar.write(f"Probabilidad: {prediccion['prob_dia']}% (Día) - {prediccion['prob_noche']}% (Noche)")
    else:
        st.sidebar.write("No hay modelo entrenado para predecir.")
else :
    ""

# ----------- MAIN APP -----------
st.title("🔍 Desaparecidos en México")
st.markdown("""
Este proyecto analiza los registros de personas desaparecidas en México con el objetivo de comprobar empíricamente la hipótesis ciudadana de que existe un repunte de casos durante los días jueves.
Confirmar la existencia de este patrón temporal demostraría que las desapariciones no son eventos aleatorios, sino que podrían estar ligadas a una dinámica criminal, rutina social o logística estructurada.

El proyecto permite subir nuevos datasets (JSON, CSV, Excel) para integrar incrementalmente la base de datos principal y reentrenar modelos de aprendizaje automático, asi como ver el analisis de los nuevos datos.
""")

if df_master.empty:
    st.warning("No hay datos en la base maestra. Por favor, sube tu primer dataset desde el panel lateral.")
else:
    stats = analizar_datos(df_master)
    
    st.header("📊 Análisis Básico a partir de los Datos")
    if 'q6' in stats:
        st.info(f"**Periodo temporal:** Del **{stats['q6']['fecha_inicio']}** al **{stats['q6']['fecha_fin']}**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Registros", value=len(df_master))
        if 'q1' in stats and 'dias' in stats['q1']:
            st.metric(label="Promedio tiempo al registro", value=f"{stats['q1']['dias']} días")
    
    with col2:
        if 'q2' in stats and 'promedio_edad' in stats['q2']:
            st.metric(label="Promedio de Edad", value=f"{stats['q2']['promedio_edad']} años")
        if 'q3' in stats and 'mayoria' in stats['q3']:
            st.metric(label="Mayoría sexo", value=stats['q3']['mayoria'])
            
    with col3:
        if 'q4' in stats:
            st.metric(label="Localidad Top 1", value=f"{stats['q4']['top1_nombre']} ({stats['q4']['top1_casos']})")
        if 'q5' in stats:
            st.metric(label="Año crítico", value=f"{stats['q5']['año']} ({stats['q5']['casos']})")

    with col4:
        if 'q7' in stats:
            st.metric(label="Día con más casos", value=f"{stats['q7']['dia_top_nombre']} ({stats['q7']['dia_top_casos']})", help="Indicador para la Hipótesis del repunte en Jueves")

    st.subheader("Exploración Visual de Datos")
    edades = df_master['edadactual'].dropna()
    if not edades.empty:
        # Streamlit Bar chart workaround for histograms, using value_counts grouped by bins
        bins = pd.cut(edades, bins=30)
        hist_df = bins.value_counts(sort=False).reset_index()
        hist_df.columns = ['Rango de Edad', 'Frecuencia']
        hist_df['Rango de Edad'] = hist_df['Rango de Edad'].astype(str)
        hist_df = hist_df.set_index('Rango de Edad')
        
        st.markdown("**Histograma: Distribución de Edad**")
        st.bar_chart(hist_df, color="#4a90e2")

    st.divider()

    # Modelos
    st.header("Análisis Avanzado e Inteligencia Artificial")
    
    st.subheader("Análisis Temporal (Regresión Lineal Simple)")
    st.markdown("Promedio de desapariciones de manera lineal.")
    res_lineal = analizar_regresion(df_master) 
    
    if res_lineal:
        col_lin1, col_lin2 = st.columns(2)
        with col_lin1:
            st.markdown("**Parte del Día**")
            st.line_chart(res_lineal['df_promedios_parte'], color="#ff7f0e")
        with col_lin2:
            st.markdown("**Día de la semana**")
            st.line_chart(res_lineal['df_promedios_dia'], color="#2ca02c")

    st.divider()
    
    st.subheader("Validación ROC-AUC (Regresión Logística)")
    st.markdown("Evaluando Capacidad de Predecir si una desaparición ocurrió de Día (0) o Noche (1).")
    st.markdown("Los nuevos datos se integran incrementalmente a la base de datos principal y se reentrena el modelo de regresión logistica que predice la hora en la que una posible desaparicion pudo ocurrir, en base a la edad, sexo, dia y mes del hecho.")
    res_logistica = analizar_regresion_logistica(df_master)
    
    col_log1, col_log2 = st.columns([2, 1])
    if res_logistica:
        with col_log1:
            st.line_chart(res_logistica['df_roc'], color=["#1f77b4", "#7f7f7f"])
        with col_log2:
            st.metric("Exactitud Inicial", f"{res_logistica['accuracy']}%")
            st.metric("Área bajo la curva (AUC)", f"{res_logistica['auc']}")
    
    st.divider()

    st.subheader("Segmentación Espacial-Temporal (K-Means Clustering)")
    st.markdown("Agrupamiento buscando similitudes geométricas en Día y Hora.")
    res_kmeans = analizar_agrupamiento(df_master)
    
    col_k1, col_k2 = st.columns([2, 1])
    if res_kmeans:
        with col_k1:
            # Native scatter chart plotting clusters color coded
            st.scatter_chart(
                res_kmeans['df_scatter'],
                x='Hora',
                y='Día (0=Lun, 6=Dom)',
                color='Cluster'
            )
        with col_k2:
            st.markdown("**Distribución de Clústeres:**")
            for cl, cnt in res_kmeans['cluster_counts'].items():
                st.markdown(f"- Clúster {cl}: **{cnt}**")

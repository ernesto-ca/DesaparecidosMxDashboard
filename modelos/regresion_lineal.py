import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def analizar_regresion(df, **kwargs):
    """
    Aplica modelos de regresión lineal para predecir y analizar tendencias en la cantidad 
    de casos de desaparición basándose en la parte del día y el día de la semana.
    """
    # Procesar y formatear la fecha y hora de los hechos a partir de las columnas existentes
    fh_str = df['fechahechos_fecha'].fillna('').str.zfill(8)
    df['dt_hechos'] = pd.to_datetime(fh_str + ' ' + df['fechahechos_hora'].fillna(''), format='%d%m%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['dt_hechos']).copy()
    
    # Extraer la hora exacta del suceso
    df['hora'] = df['dt_hechos'].dt.hour
    
    # Categorizar las horas en 4 partes del día diferentes (Madrugada, Mañana, Tarde, Noche)
    bins = [-1, 5, 11, 18, 24]
    labels = [0, 1, 2, 3] 
    df['parte_dia'] = pd.cut(df['hora'], bins=bins, labels=labels)
    
    # Extraer el identificar estadístico del día en la semana y también una fecha limpia (solo día, sin hora)
    df['dia_semana'] = df['dt_hechos'].dt.dayofweek
    df['fecha_solo'] = df['dt_hechos'].dt.date
    
    # Primer análisis por modelo: Regresión lineal por partes del día
    casos_por_parte = df.groupby(['fecha_solo', 'parte_dia'], observed=True).size().reset_index(name='casos')
    X_parte = casos_por_parte[['parte_dia']].values.astype(int)
    y_parte = casos_por_parte['casos'].values.astype(float)
    
    # Entrenar modelo con la parte del día
    modelo_parte = LinearRegression()
    modelo_parte.fit(X_parte, y_parte)
    
    nombres_partes = {0: 'Madrugada', 1: 'Mañana', 2: 'Tarde', 3: 'Noche'}

    # Segundo análisis por modelo: Regresión lineal por días de la semana
    casos_por_dia = df.groupby(['fecha_solo', 'dia_semana'], observed=True).size().reset_index(name='casos')
    X_dia = casos_por_dia[['dia_semana']].values.astype(float)
    y_dia = casos_por_dia['casos'].values.astype(float)
    
    # Entrenar modelo considerando el día de la semana
    modelo_dia = LinearRegression()
    modelo_dia.fit(X_dia, y_dia)
    
    # Extraer métricas resumen de los casos agrupados por día de la semana
    promedio_dias = casos_por_dia.groupby('dia_semana', observed=True)['casos'].mean()
    nombres_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    max_dia = int(promedio_dias.idxmax())

    # Agrupar datos de salida sobre partes del día para visualización directa
    partes_ordenadas = [nombres_partes[i] for i in range(4)]
    df_parte = pd.DataFrame({
        'parteDia': partes_ordenadas,
        'promedioCasos': [casos_por_parte[casos_por_parte['parte_dia'] == i]['casos'].mean() for i in range(4)]
    })
    # Asegurar orden cronológico para evitar que Streamlit ordene alfabéticamente
    df_parte['parteDia'] = pd.Categorical(df_parte['parteDia'], categories=partes_ordenadas, ordered=True)
    df_parte = df_parte.set_index('parteDia')
    
    # Agrupar datos de salida sobre días para visualización
    df_dia = pd.DataFrame({
        'dia': nombres_dias,
        'promedioCasos': [casos_por_dia[casos_por_dia['dia_semana'] == i]['casos'].mean() for i in range(7)]
    })
    # Asegurar orden de los días de la semana
    df_dia['dia'] = pd.Categorical(df_dia['dia'], categories=nombres_dias, ordered=True)
    df_dia = df_dia.set_index('dia')

    return {
        'df_promedios_parte': df_parte,
        'df_promedios_dia': df_dia,
        'max_promedio_dia': nombres_dias[max_dia],
        'jueves_promedio': round(promedio_dias[3], 2) if 3 in promedio_dias else 0,
        'coef': round(modelo_dia.coef_[0], 4)
    }

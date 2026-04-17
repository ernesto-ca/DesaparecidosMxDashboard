import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

def analizar_agrupamiento(df, **kwargs):
    """
    Realiza un análisis de agrupamiento (clustering) utilizando el algoritmo K-Means.
    Extrae la fecha y hora de los hechos, prepara las características y agrupa los incidentes en clusters.
    """
    # Rellenar con ceros a la izquierda la fecha y combinar con la hora para crear un datetime
    fh_str = df['fechahechos_fecha'].fillna('').str.zfill(8)
    df['dt_hechos'] = pd.to_datetime(fh_str + ' ' + df['fechahechos_hora'].fillna(''), format='%d%m%Y %H:%M:%S', errors='coerce')
    
    # Eliminar registros sin fecha de hechos o sin municipio
    df = df.dropna(subset=['dt_hechos', 'municipio']).copy()
    
    # Extraer componentes temporales que nos ayuden a encontrar patrones
    df['hora'] = df['dt_hechos'].dt.hour
    df['dia_semana'] = df['dt_hechos'].dt.dayofweek
    df['mes'] = df['dt_hechos'].dt.month
    
    # Codificar la variable categórica de texto 'municipio' a valores numéricos
    le = LabelEncoder()
    df['municipio_encoded'] = le.fit_transform(df['municipio'].astype(str))
    
    # Seleccionar las variables (características) a utilizar en el algoritmo KMeans
    caracteristicas = ['municipio_encoded', 'hora', 'dia_semana']
    X = df[caracteristicas]
    
    # Escalar los datos para que todas estas variables numéricas estén en la misma escala
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Inicializar y entrenar el modelo K-Means para encontrar 3 grupos
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Contar la cantidad de registros que terminaron en cada cluster
    cluster_counts = df['cluster'].value_counts().to_dict()

    # Preparar un DataFrame para su visualización en gráfico de dispersión,
    # añadiendo "jitter" aleatorio (dispersión artificial) para visibilizar puntos encimados
    df_scatter = pd.DataFrame({
        'Hora': df['hora'] + np.random.uniform(-0.3, 0.3, size=len(df)),
        'Día (0=Lun, 6=Dom)': df['dia_semana'] + np.random.uniform(-0.3, 0.3, size=len(df)),
        'Cluster': df['cluster'].astype(str)
    })

    return {
        'df_scatter': df_scatter,
        'n_clusters': n_clusters,
        'cluster_counts': {str(k): v for k, v in cluster_counts.items()}
    }

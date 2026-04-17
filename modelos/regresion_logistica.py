import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

current_script_dir = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(current_script_dir, 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_LOGISTICA_PATH = os.path.join(MODEL_DIR, 'sgd_logistica_roc.joblib')
MODEL_SCALER_PATH = os.path.join(MODEL_DIR, 'sgd_logistica_scaler.joblib')

def analizar_regresion_logistica(df, is_incremental=False, **kwargs):
    """
    Entrena y evalúa un modelo de regresión logística estocástica (SGD) 
    para predecir si una desaparición ocurre de noche basado en sexo, edad y momento del hecho.
    Este modelo evalúa datos incrementalmente en vez de todos de golpe si is_incremental es verdadero.
    """
    # Procesar fecha y limpiarla asegurando el formato estándar y manejando valores faltantes
    fh_str = df['fechahechos_fecha'].fillna('').str.zfill(8)
    df['dt_hechos'] = pd.to_datetime(fh_str + ' ' + df['fechahechos_hora'].fillna(''), format='%d%m%Y %H:%M:%S', errors='coerce')
    
    # Descartar filas y registros con información crítica faltante
    df = df.dropna(subset=['dt_hechos', 'edadactual', 'sexo']).copy()
    
    # Extraer piezas informativas de relevancia sobre el horario
    df['hora'] = df['dt_hechos'].dt.hour
    df['dia_semana'] = df['dt_hechos'].dt.dayofweek
    df['mes'] = df['dt_hechos'].dt.month
    
    # Variable a predecir (Objetivo): Clasificación binaria entre Día (0) frente a Noche (1)
    df['ocurre_de_noche'] = df['hora'].apply(lambda x: 1 if (x >= 19 or x < 6) else 0)
    
    # Variables de entrada y salida respectivamente
    caracteristicas = ['sexo', 'edadactual', 'dia_semana', 'mes']
    X = df[caracteristicas]
    y = df['ocurre_de_noche']
    
    # Normalizar (escalar) los valores de entrada para el entrenamiento apropiado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Guardar el scaler para poder reutilizarlo en predicciones individuales
    joblib.dump(scaler, MODEL_SCALER_PATH)
    
    classes = np.array([0, 1])
    
    # Implementación opcional para entrenar incrementalmente a partir del modelo anterior guardado
    if is_incremental and os.path.exists(MODEL_LOGISTICA_PATH):
        modelo = joblib.load(MODEL_LOGISTICA_PATH)
        modelo.partial_fit(X_scaled, y, classes=classes)
    else:
        # Calcular pesos de manera equilibrada para compensar cualquier sesgo de clasificaciones comunes
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights_dict = {c: w for c, w in zip(classes, weights)}
        
        # Empezar un clasificador estocástico de descenso de gradiente para hacer regresión logística
        modelo = SGDClassifier(loss='log_loss', class_weight=class_weights_dict, random_state=42, max_iter=1000)
        modelo.partial_fit(X_scaled, y, classes=classes)
        
    # Guardar el modelo entrenado serializado en disco
    joblib.dump(modelo, MODEL_LOGISTICA_PATH)
    
    # Generar predicciones concretas e índices de probabilidad con el modelo
    y_pred = modelo.predict(X_scaled)
    y_proba = modelo.predict_proba(X_scaled)[:, 1]
    
    # Calcular métricas de calificación clave: exactitud general y trazado de las curvas ROC de evaluación
    accuracy = accuracy_score(y, y_pred)
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Preparar el DataFrame empaquetando las valoraciones relativas de falsos y verdaderos positivos
    df_roc = pd.DataFrame({
        'Tasa Falsos Positivos (FPR)': fpr,
        'ROC Curve': tpr,
        'Azar (Referencia)': fpr
    }).set_index('Tasa Falsos Positivos (FPR)')

    return {
        'df_roc': df_roc,
        'accuracy': round(accuracy * 100, 2),
        'auc': round(roc_auc, 2)
    }


def predecir_hora_desaparicion(sexo: int, edad: int, dia_semana: int, mes: int) -> dict | None:
    """
    Parámetros:
        sexo       : 0 = Mujer, 1 = Hombre (codificado igual que en el preprocesamiento)
        edad       : Edad actual de la persona
        dia_semana : 0 = Lunes … 6 = Domingo
        mes        : 1 = Enero … 12 = Diciembre

    Retorna un diccionario con la predicción o None si el modelo aún no está entrenado.
    """
    # Verificar que existan tanto el modelo como el escalador guardados en disco
    if not os.path.exists(MODEL_LOGISTICA_PATH) or not os.path.exists(MODEL_SCALER_PATH):
        return None

    # Cargar modelo y escalador previamente entrenados
    modelo = joblib.load(MODEL_LOGISTICA_PATH)
    scaler = joblib.load(MODEL_SCALER_PATH)

    # Construir el vector de características del caso a predecir
    X_nuevo = np.array([[sexo, edad, dia_semana, mes]], dtype=float)
    X_nuevo_scaled = scaler.transform(X_nuevo)

    # Obtener la clase predicha (0 = Día, 1 = Noche) y las probabilidades
    clase_predicha = modelo.predict(X_nuevo_scaled)[0]
    probabilidades = modelo.predict_proba(X_nuevo_scaled)[0]  # [prob_dia, prob_noche]

    # Estimar un rango horario representativo según la clase predecida
    if clase_predicha == 1:  # Noche: 19:00 – 05:59
        hora_estimada = "19:00 – 05:59 (Noche)"
    else:                    # Día: 06:00 – 18:59
        hora_estimada = "06:00 – 18:59 (Día)"

    return {
        'clase': int(clase_predicha),
        'etiqueta': hora_estimada,
        'prob_dia': round(float(probabilidades[0]) * 100, 1),
        'prob_noche': round(float(probabilidades[1]) * 100, 1),
    }

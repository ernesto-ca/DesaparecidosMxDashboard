import pandas as pd
import os

current_script_dir = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(current_script_dir, '..', 'data', 'desaparecidos_preprocesado.csv')

def analizar_datos(df):
    if df is None or df.empty:
        return {}
    
    resultados = {}

    # 1. Promedio de tiempo
    try:
        fh_str = df['fechahechos_fecha'].fillna('')
        fc_str = df['fechacaptura_fecha'].fillna('')
        df['dt_hechos'] = pd.to_datetime(fh_str + ' ' + df['fechahechos_hora'].fillna(''), format='%d%m%Y %H:%M:%S', errors='coerce')
        df['dt_captura'] = pd.to_datetime(fc_str + ' ' + df['fechacaptura_hora'].fillna(''), format='%d%m%Y %H:%M:%S', errors='coerce')
        
        tiempo_registro = (df['dt_captura'] - df['dt_hechos']).dt.total_seconds() / 86400
        promedio_dias_registro = tiempo_registro.mean()
        resultados['q1'] = {'dias': round(promedio_dias_registro, 2)}
    except Exception as e:
        resultados['q1'] = {'error': str(e)}

    # 2. Promedio edad
    if 'edadactual' in df.columns:
        edad_numerica = pd.to_numeric(df['edadactual'], errors='coerce')
        resultados['q2'] = {'promedio_edad': round(edad_numerica.mean(), 2)}

    # 3. Sexo
    if 'sexo' in df.columns:
        conteos_sexo = df['sexo'].value_counts(normalize=True) * 100
        conteos_absolutos = df['sexo'].value_counts()
        labels_sexo = {0: 'Hombres', 1: 'Mujeres'}
        mayoria_sexo_id = conteos_absolutos.idxmax()
        
        detalles_sexo = []
        for val, prop in conteos_sexo.items():
            detalles_sexo.append({
                'nombre': labels_sexo.get(val, 'Desconocido'),
                'casos': int(conteos_absolutos[val]),
                'pct': round(prop, 2)
            })
            
        resultados['q3'] = {
            'mayoria': labels_sexo.get(mayoria_sexo_id, 'Desconocido').upper(),
            'detalles': detalles_sexo
        }

    # 4. Localidad
    if 'municipio' in df.columns:
        top_municipio = df['municipio'].value_counts()
        if not top_municipio.empty:
            top3 = [{'mun': m, 'casos': int(c)} for m, c in list(top_municipio.items())[1:4]]
            resultados['q4'] = {
                'top1_nombre': top_municipio.index[0],
                'top1_casos': int(top_municipio.iloc[0]),
                'top3_adicional': top3
            }

    # 5. Año con más desapariciones
    if 'dt_hechos' in df.columns:
        df['año_hechos'] = df['dt_hechos'].dt.year
        año_top = df['año_hechos'].value_counts()
        if not año_top.empty:
            max_año = int(año_top.index[0])
            total_año_max = int(año_top.iloc[0])
            
            df_año_top = df[df['año_hechos'] == max_año]
            pct_sexo_año = df_año_top['sexo'].value_counts(normalize=True) * 100 if 'sexo' in df_año_top.columns else {}
            dist_sexo = [{'nombre': labels_sexo.get(v, 'Desc'), 'pct': round(p, 2)} for v, p in pct_sexo_año.items()]
            
            resultados['q5'] = {
                'año': max_año,
                'casos': total_año_max,
                'distribucion_sexo': dist_sexo
            }

    # 6. Rango de tiempo del dataset
    if 'dt_hechos' in df.columns:
        dt_valida = df['dt_hechos'].dropna()
        if not dt_valida.empty:
            min_date = dt_valida.min().strftime('%Y-%m-%d')
            max_date = dt_valida.max().strftime('%Y-%m-%d')
            resultados['q6'] = {
                'fecha_inicio': min_date,
                'fecha_fin': max_date
            }

    # 7. Día con más desapariciones (Hipótesis del Jueves)
    if 'dt_hechos' in df.columns:
        dt_valida = df['dt_hechos'].dropna()
        if not dt_valida.empty:
            conteo_dias = dt_valida.dt.dayofweek.value_counts()
            if not conteo_dias.empty:
                max_dia_idx = conteo_dias.idxmax()
                mapa_dias = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
                resultados['q7'] = {
                    'dia_top_nombre': mapa_dias.get(max_dia_idx, 'Desconocido'),
                    'dia_top_casos': int(conteo_dias.iloc[0])
                }

    return resultados



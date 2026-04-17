import pandas as pd

def preprocesar_dataframe(df):
    if df is None or df.empty:
        return df

    # Convertir columnas a minúsculas
    df.columns = df.columns.str.lower()

    columnas_a_eliminar = [
        'publicarficha', 'idvinculacion', 'idvictimadirecta', 
        'iddependenciaorigen', 'cantidadregistros', 'estatusvictimadirecta', 'idreporte'
    ]
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')
    
    # Convertir edadactual a numérico
    df['edadactual'] = pd.to_numeric(df['edadactual'], errors='coerce')
    
    # Filtra y mantiene solo las filas donde NINGUNA columna contenga 'CONFIDENCIAL'
    df = df[~df.apply(lambda row: row.astype(str).str.contains('CONFIDENCIAL', case=False).any(), axis=1)]
    
    # Filtra y mantiene solo las filas donde NINGUNA columna contenga la palabra exacta 'SIN'
    df = df[~df.apply(lambda row: row.astype(str).str.contains(r'\bSIN\b', case=False, regex=True).any(), axis=1)]

    # Eliminar filas con valores nulos
    df.dropna(subset=['sexo', 'nombre', 'fechahechos', 'fechapercato', 'edadactual'], inplace=True)


    for col in ['fechahechos', 'fechapercato', "fechacaptura"]:
        if col in df.columns:
            dt_col = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_fecha'] = dt_col.dt.strftime('%d%m%Y')
            df[f'{col}_hora'] = dt_col.dt.strftime('%H:%M:%S')
            df = df.drop(columns=[col])

    cols_nombre = ['nombre', 'primerapellido', 'segundoapellido']
    for c in cols_nombre:
        if c not in df.columns:
            df[c] = ''
    
    df['nombre_completo'] = (
        df['nombre'].fillna('').astype(str) + " " +
        df['primerapellido'].fillna('').astype(str) + " " +
        df['segundoapellido'].fillna('').astype(str)
    )
    df['nombre_completo'] = df['nombre_completo'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df.drop(columns=cols_nombre)

    # Mapea 'HOMBRE' -> 0, 'MUJER' -> 1
    df['sexo'] = df['sexo'].map({'HOMBRE': 0, 'MUJER': 1})

    return df

# Análisis de Desaparecidos en México (Dashboard Interactivo)

Una aplicación web interactiva (dashboard) para analizar, visualizar y aplicar modelos de inteligencia artificial a la información sobre casos de personas desaparecidas en México.

La meta de este proyecto es que a través de una herramienta profesional y fácil de usar, los usuarios puedan subir sus bases de datos y obtener instantáneamente el procesamiento, las estadísticas y la ejecución de modelos matemáticos para descubrir patrones. Asi mismo, se busca comprobar empíricamente la hipótesis ciudadana de que existe un repunte de casos durante los días jueves.

## DEMO

![demostracion](demo/desaparecidos-mx-dashboard.gif)

## Estructura del Proyecto: Archivos y Modelos


### Archivos Principales y Scripts (Procesamiento de Datos)

*   **`streamlit_app.py`**: Es el **corazón de la página web**. Al ejecutar este archivo, se abre el panel interactivo (el "dashboard") donde puedes subir un archivo con datos, hacer clics en botones y ver todas las gráficas y resultados en tu navegador de internet.
*   **`scripts/conversor_json_csv.py`**: A veces la información se descarga en formatos que son difíciles de leer (como JSON). Este programa toma ese formato y lo convierte en una tabla normal (CSV), como si fuera una hoja de cálculo en Excel.
*   **`scripts/preprocesamiento.py`**: Se encarga de la **limpieza de los datos**. Quita las columnas que no necesitamos, arregla el formato de las fechas (separando el día de la hora) y une el nombre completo de las personas. Es como "limpiar los ingredientes" antes de cocinar, asegurando que la información esté lista para ser analizada por la computadora sin errores.
*   **`scripts/analisis.py`**: Realiza las **cuentas iniciales y estadísticas matemáticas**, además de preparar la información que será transformada en imágenes como histogramas o gráficas, para que se entiendan en un solo vistazo.

### Modelos de Inteligencia Artificial

Los siguientes archivos usan matemáticas e inteligencia artificial ("Machine Learning") para ir más allá de los porcentajes, permitiéndonos predecir o encontrar patrones invisibles a simple vista:

*   **`modelos/regresion_lineal.py`** (Regresión Lineal): En los datos, pude existir mucha varianza y en la grafica se ve reflejada en la dispersión de los puntos. Este modelo traza la "mejor línea recta" que pasa en medio de todos ellos, para decirnos si existe alguna tendencia ascendente o descendente entre ciertos números (por ejemplo, ver cómo evolucionan los casos con el tiempo).
*   **`modelos/agrupamiento.py`** (Agrupamiento / K-Means):Dentro de los datos existen patrones entre ellos, pero estan desordenados es por eso que el mdoelo utiliza grupos para separarlos y encontrar patrones, zonas o grupos de víctimas afines.
*   **`modelos/regresion_logistica.py`** (Regresión Logística): A pesar de llamarse regresión, se usa para **clasificar y calcular probabilidades**. En este proyecto en específico, analiza si las desapariciones se concentran en ciertos días, por ejemplo, ayudando a probar la hipótesis de si las desapariciones *aumentan los días jueves*. Evalúa si sus propias predicciones son buenas midiendo su exactitud, asi como tambien intentar predecir si una persona desaparecida que oslo se sabe su sexo, eda, dia y mes de desaparición, es probable que haya desaparecido en la noche o el día.

## Instalación y Ejecución

Para facilitar el uso del sistema, se incluyó un archivo de configuración llamado `requirements.txt`. Aquí están los pasos para usarlo incluso si eres principiante:

1. **Instalar los requisitos mínimos**: Abre una terminal o ventana de comandos y, estando en la carpeta de este proyecto, copia y pega el siguiente comando:
   ```bash
   pip install -r requirements.txt
   ```
2. **Ejecutar la plataforma web**: Una vez terminado el paso anterior, inicia el "dashboard" escribiendo:
   ```bash
   streamlit run streamlit_app.py
   ```
   *Se abrirá automáticamente una pestaña nueva en tu navegador de internet lista para usarse.*

---

## Referencias

*   **Fuentes de Datos Gubernamentales**:
    * Registro Nacional de Personas Desaparecidas y No Localizadas (RNPDNO) - Secretaría de Gobernación: [https://consultapublicarnpdno.segob.gob.mx/consulta](https://consultapublicarnpdno.segob.gob.mx/consulta)
*   **Tecnologías de Dashboarding**:
    * Documentación y recursos de Streamlit (la herramienta usada para construir la web): [https://streamlit.io](https://streamlit.io)

---

## Autor

*   **Autor:** Ernesto Cabañas Albarran
*   **Fecha de última modificación:** 16 de abril de 2026

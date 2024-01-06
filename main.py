from datasets import load_dataset
import numpy as np
import pandas as pd #PT2
import requests #PT4
import sys #PT6
import matplotlib.pyplot as plt #PT7
import seaborn as sns #PT7
from sklearn.manifold import TSNE
import plotly.express as px #PT9
import plotly.graph_objects as go #PT9
from sklearn.model_selection import train_test_split #PT10
from sklearn.linear_model import LinearRegression #PT10
from sklearn.metrics import mean_squared_error #PT10



def separate():
    print('''--------------------------------------------------------------------------------------------------------------------------------------------------''')

def separador(func):
    def wrapper(*args, **kwargs):
        print("===============================================================================================================================================")
        func(*args, **kwargs)
        print("===============================================================================================================================================")
    return wrapper


@separador
def cargar_dataset_proyecto_pt1():
    print(' --- PROYECTO INTEGRADOR PARTE 1 --- ')
    dataset = load_dataset("mstz/heart_failure")
    data = dataset["train"]

    lista_edades = data['age']
    arreglo_edades = np.array(lista_edades)
    promedio_edades = round(np.mean(lista_edades), 2)

    print(f"Promedio de edad de los participantes: {promedio_edades} años")


@separador
def procesar_dataframe_proyecto_pt2(df):
    print(' --- PROYECTO INTEGRADOR PARTE 2 --- ')
    df_is_dead = None
    df_isnot_dead = None
    df = pd.DataFrame(df)
    print(f"El Dataframe para este proyecto es: \n {df}")

    palabras_clave_muerte = ['death', 'dead']
    columna_muerte = next((col for col in df.columns if any(keyword in col.lower() for keyword in palabras_clave_muerte)), None)    
    if columna_muerte is None:
        print("No se encontró una columna relacionada con la muerte ('DEATH_EVENT' o similar).")
    elif columna_muerte in df.columns:
        df_is_dead = df[df[columna_muerte] == 1]
        df_isnot_dead = df[df[columna_muerte] == 0]
    
    separate()

    print(f"Dataframe de las personas que fallecieron: \n {df_is_dead}")
    separate()

    print(f"Dataframe de las personas que NO fallecieron: \n {df_isnot_dead}")
    separate()

    if df_is_dead is not None and df_isnot_dead is not None:
        promedio_edad_fallecido = round(np.mean(df_is_dead['age']), 2)
        promedio_edad_nofallecido = round(np.mean(df_isnot_dead['age']), 2)
        print(f"Promedio de edad de los fallecidos: {promedio_edad_fallecido} años")
        print(f"Promedio de edad de los que NO fallecieron: {promedio_edad_nofallecido} años")


@separador
def procesar_dataframe_proyecto_pt3(dataframe):
    print(' --- PROYECTO INTEGRADOR PARTE 3 --- ')

    palabras_clave_muerte = ['death', 'dead']  

    tipo_de_datos = dataframe.dtypes
    print(f"El tipo de dato de cada columna es: \n", tipo_de_datos)
    separate()

    columna_muerte = next((col for col in dataframe.columns if any(keyword in col.lower() for keyword in palabras_clave_muerte)), None)
    
    if columna_muerte is None:
        print("No se encontró una columna relacionada con la muerte ('DEATH_EVENT' o similar).")
    else:
        # Convertir la columna de muerte a tipo booleano
        dataframe[columna_muerte] = dataframe[columna_muerte].astype(bool)

        tipo_de_datos = dataframe.dtypes
        print(f"Después de la correción, el tipo de dato de cada columna es: \n", tipo_de_datos)
        separate()

        print(f"El nuevo dataframe se ve así: \n {dataframe}")
        separate()

        cuenta_hombre_fumador = dataframe[(dataframe['smoking'] == 1) & (dataframe['sex'] == 1)].shape[0]
        cuenta_mujer_fumadora = dataframe[(dataframe['smoking'] == 1) & (dataframe['sex'] == 0)].shape[0]
        proporcion_mujer_x_hombre = cuenta_mujer_fumadora / cuenta_hombre_fumador if cuenta_hombre_fumador != 0 else 0
        print(f"La cantidad de hombres fumadores es: {cuenta_hombre_fumador}")
        print(f"La cantidad de mujeres fumadoras es: {cuenta_mujer_fumadora}")
        print(f"La proporción de fumadoras por cada fumador es: {proporcion_mujer_x_hombre:.2f}")


@separador
def descargar_datos_proyecto_pt4(url, salida_archivo):
    print(' --- PROYECTO INTEGRADOR PARTE 4 --- ')

    response = requests.get(url)

    if response.status_code == 200:
        with open(salida_archivo, 'w', encoding='utf-8') as file:
            file.write(response.text)
        print(f"Los datos se han guardado en {salida_archivo}")
    else:
        print(f"No se pudo almacenar los datos. Código de respuesta: {response.status_code}")


@separador
def procesar_dataframe_proyecto_pt5(df: pd.DataFrame) -> None:
    print(' --- PROYECTO INTEGRADOR PARTE 5 --- ')

    if df.isnull().values.any():
        print("Hay valores faltantes. Se eliminarán las filas con valores faltantes")
        df = df.dropna()  

    if df.duplicated().any():
        print("Existen filas duplicadas en el DataFrame. Se eliminarán los duplicados")
        df = df.drop_duplicates()  

    Q1 = df['age'].quantile(0.25)
    Q3 = df['age'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['age'] >= Q1 - 1.5 * IQR) & (df['age'] <= Q3 + 1.5 * IQR)]

    bins = [0, 12, 19, 39, 59, float('inf')]
    labels = ['Niño', 'Adolescente', 'Adulto joven', 'Adulto', 'Adulto mayor']
    df['edad_categorizada'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    resultado_csv = "resultado_proyecto_int_PT5.csv"
    df.to_csv(resultado_csv, index=False)
    print(f"El resultado se ha guardado en {resultado_csv}")


@separador
def procesar_dataframe_proyecto_pt6(url):
    print(' --- PROYECTO INTEGRADOR PARTE 6 --- ')
    # Descargar datos
    descargar_datos_proyecto_pt4(url, "nuevos_datos.csv")
    
    # Cargar DataFrame
    nuevos_datos = pd.read_csv("nuevos_datos.csv")
    
    # Llamar funciones de las partes anteriores
    procesar_dataframe_proyecto_pt2(nuevos_datos)
    procesar_dataframe_proyecto_pt3(nuevos_datos)
    procesar_dataframe_proyecto_pt5(nuevos_datos)

#Llamadas a las funciones
cargar_dataset_proyecto_pt1()

#Ejecución de la parte 6 del proyecto (procesar_dataframe_proyecto_pt6)
if __name__ == "__main__":
    #Solicita la URL al usuario por la consola
    url_descarga_pt6 = sys.argv[1] if len(sys.argv) > 1 else input("Por favor, ingrese la URL de los datos: ")
    procesar_dataframe_proyecto_pt6(url_descarga_pt6)    
    resultado = pd.read_csv("resultado_proyecto_int_PT5.csv")

    # 1. Gráfica de distribución de edades
    print("--- PROYECTO PT7: GRÁFICA DE DISTRIBUCIÓN DE EDADES ---")
    plt.figure(figsize=(10, 6))
    plt.hist(resultado['age'], bins=6, color='skyblue', edgecolor='black')
    plt.title('Distribución de Edades')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # 2. Histogramas agrupados por hombre / mujer
    print("--- PROYECTO PT7: HISTOGRAMAS AGRUPADOS POR HOMBRE Y MUJER ---")
    # Filtrar datos por género
    data_hombres = resultado[resultado['sex'] == 1]
    data_mujeres = resultado[resultado['sex'] == 0]

    # Configurar la figura
    plt.figure(figsize=(15, 10))

    # Anémicos
    plt.bar([-1, -0.2], [data_hombres['anaemia'].sum(), data_mujeres['anaemia'].sum()], color=['blue', 'red'], width=0.8, label=['Hombres', 'Mujeres'])
    # Diabéticos
    plt.bar([1.2, 2], [data_hombres['diabetes'].sum(), data_mujeres['diabetes'].sum()], color=['blue', 'red'], width=0.8, label=['Hombres', 'Mujeres'])
    # Fumadores
    plt.bar([3.4, 4.2], [data_hombres['smoking'].sum(), data_mujeres['smoking'].sum()], color=['blue', 'red'], width=0.8, label=['Hombres', 'Mujeres'])
    # Muertos
    plt.bar([5.6, 6.4], [data_hombres['DEATH_EVENT'].sum(), data_mujeres['DEATH_EVENT'].sum()], color=['blue', 'red'], width=0.8, label=['Hombres', 'Mujeres'])

    plt.title('Cantidad por categoría y género')
    plt.xlabel('Categoría')
    plt.ylabel('Cantidad')
    plt.xticks([-0.6, 1.6, 3.8, 6], ['Anémicos', 'Diabéticos', 'Fumadores', 'Muertos'])
    plt.legend()

    plt.show()


    print("--- PROYECTO PT8: GRÁFICOS DE TORTA / DISTRIBUCIONES ---")
    # Mapeo de valores
    etiqueta_si_no = {0: 'No', 1: 'Sí'}
    etiqueta_si_no_TF = {False: 'No', True: 'Sí'}

    # Reemplazar valores en las columnas
    resultado['anaemia'] = resultado['anaemia'].map(etiqueta_si_no)
    resultado['diabetes'] = resultado['diabetes'].map(etiqueta_si_no)
    resultado['smoking'] = resultado['smoking'].map(etiqueta_si_no)
    resultado['DEATH_EVENT'] = resultado['DEATH_EVENT'].map(etiqueta_si_no_TF)

    # Filtrar datos por categoría
    cantidad_anemicos = resultado['anaemia'].value_counts()
    cantidad_diabeticos = resultado['diabetes'].value_counts()
    cantidad_fumadores = resultado['smoking'].value_counts()
    cantidad_muertos = resultado['DEATH_EVENT'].value_counts()

    # Configurar la figura con subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Gráfica de torta para anémicos
    axs[0, 0].pie(cantidad_anemicos, labels=cantidad_anemicos.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'plum'])
    axs[0, 0].set_title('Anémicos')

    # Gráfica de torta para diabéticos
    axs[0, 1].pie(cantidad_diabeticos, labels=cantidad_diabeticos.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'plum'])
    axs[0, 1].set_title('Diabéticos')

    # Gráfica de torta para fumadores
    axs[1, 0].pie(cantidad_fumadores, labels=cantidad_fumadores.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'plum'])
    axs[1, 0].set_title('Fumadores')

    # Gráfica de torta para muertos
    axs[1, 1].pie(cantidad_muertos, labels=cantidad_muertos.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'plum'])
    axs[1, 1].set_title('Muertos')

    plt.tight_layout()
    plt.show()

    print("--- PROYECTO PT9: GRÁFICO DE DISPERSIÓN 3D ---")
    # Eliminar columnas no necesarias
    X = resultado.drop(columns=['DEATH_EVENT', 'edad_categorizada']).copy()

    # Convertir columnas categóricas a variables dummy
    X = pd.get_dummies(X, columns=['anaemia', 'diabetes', 'smoking'], drop_first=True)

    # Exportar un array unidimensional de la columna objetivo
    y = resultado['DEATH_EVENT'].values

    # Realizar la reducción de dimensionalidad con t-SNE
    X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(X)

    # Crear un DataFrame con los resultados de t-SNE y la columna objetivo
    df_tsne = pd.DataFrame({'Eje_X': X_embedded[:, 0], 'Eje_Y': X_embedded[:, 1], 'Eje_Z': X_embedded[:, 2], 'DEATH_EVENT': y})

    # Mapear colores a las clases
    colors = {'No': 'blue', 'Sí': 'red'}
    df_tsne['color'] = df_tsne['DEATH_EVENT'].map(colors)

    # Crear la figura
    fig = go.Figure()

    # Añadir el trazado de dispersión 3D a la figura
    fig.add_trace(go.Scatter3d(
        x=df_tsne['Eje_X'], y=df_tsne['Eje_Y'], z=df_tsne['Eje_Z'],
        mode='markers',  # Estilo de marcador
        marker=dict(
            size=5,  # Tamaño de los marcadores
            color=df_tsne['color'],  # Usar la nueva columna 'color' para la escala de colores
            opacity=0.8  # Opacidad de los marcadores
        )
    ))

    # Personalizar el diseño de la gráfica
    fig.update_layout(
        title='Parte 9: Analizando Distribuciones 3',  # Título de la gráfica
        scene=dict(
            xaxis_title='Eje_X',  # Etiqueta del eje x
            yaxis_title='Eje_Y',  # Etiqueta del eje y
            zaxis_title='Eje_Z'  # Etiqueta del eje z
        )
    )
    fig.show()

    
    print("--- PROYECTO PT10: Prediciendo Datos de una Columna")

    # 1. Eliminar las columnas DEATH_EVENT, age y dad_categorizada
    X = resultado.drop(columns=['DEATH_EVENT', 'age', 'edad_categorizada']).copy()
    X = pd.get_dummies(X, columns=['anaemia', 'diabetes', 'smoking'], drop_first=True)

    # 2.Columna 'age' como el vector Y
    y = resultado['age'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2.1. Ajustar una regresión lineal
    modelo_regresion = LinearRegression()
    modelo_regresion.fit(X_train, y_train)

    # 3. Predice las edades y compara
    y_pred = modelo_regresion.predict(X_test)
    resultados = pd.DataFrame({'Edad Real': y_test, 'Edad Predicha': y_pred})

    # 4. Calcular el error cuadrático medio
    mse = mean_squared_error(y_test, y_pred)

    print("Resultados de la Regresión Lineal:")
    print(resultados)
    print("\nError Cuadrático Medio:", mse)
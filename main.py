from datasets import load_dataset
import numpy as np
import pandas as pd #PT2
import requests #PT4
import sys


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

    palabras_clave_muerte = ['death', 'dead']  # Agregar esta línea

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
    procesar_dataframe_proyecto_pt6(url_descarga_pt6)    #Procesa los nuevos datos obtenidos de la URL

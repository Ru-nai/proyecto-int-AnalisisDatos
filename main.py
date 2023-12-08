from datasets import load_dataset
import numpy as np
import pandas as pd #PT2
import requests #PT4

def separador():
    print("===============================================================================================================================================")

print(' --- PROYECTO INTEGRADOR PARTE 1 --- ')
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

lista_edades = data['age']
arreglo_edades = np.array(lista_edades)
promedio_edades = round(np.mean(lista_edades), 2)

print(f"Promedio de edad de los participantes: {promedio_edades} años")
separador()
separador()

print(' --- PROYECTO INTEGRADOR PARTE 2 --- ')

#1. Convertir la estructura Dataset en un DataFrame de Pandas usando pd.DataFrame.
df = pd.DataFrame(data)
print(f"El Dataframe para este proyecto es: \n {df}")

#2. Separar el dataframe en dos diferentes, uno conteniendo las filas con personas que perecieron (is_dead=1) y otro con el complemento.
df_is_dead = df[df['is_dead'] == 1]
df_isnot_dead = df[df['is_dead'] == 0]
separador()
separador()

print(f"Dataframe de las personas que fallecieron: \n {df_is_dead}")
separador()
separador()
print(f"Dataframe de las personas que NO fallecieron: \n {df_isnot_dead}")
separador()
separador()

#3. Calcular los promedios de las edades de cada dataset e imprimir.
promedio_edad_fallecido = round(np.mean(df_is_dead['age']), 2)
promedio_edad_nofallecido = round(np.mean(df_isnot_dead['age']), 2)
print(f"Promedio de edad de los fallecidos: {promedio_edad_fallecido} años")
print(f"Promedio de edad de los que NO fallecieron: {promedio_edad_nofallecido} años")
separador()
separador()

print(' --- PROYECTO INTEGRADOR PARTE 3 --- ')
#1. Verificar que los tipos de datos son correctos en cada colúmna (por ejemplo que no existan colúmnas numéricas en formato de cadena).
tipo_de_datos = df.dtypes
print(f"El tipo de dato de cada columna es: \n", tipo_de_datos)
separador()
separador()

df['is_dead'] = df['is_dead'].astype(bool)
tipo_de_datos = df.dtypes
print(f"Después de la correción, el tipo de dato de cada columna es: \n", tipo_de_datos)
separador()
separador()

print(f"El nuevo dataframe se ve así: \n {df}")
separador()
separador()

#2. Calcular la cantidad de hombres fumadores vs mujeres fumadoras (usando agregaciones en Pandas).
cuenta_hombre_fumador = df[(df['is_smoker'] == True) & (df['is_male'] == True)].shape[0]
cuenta_mujer_fumador = df[(df['is_smoker'] == True) & (df['is_male'] == False)].shape[0]
proporcion_mujer_x_hombre = cuenta_mujer_fumador / cuenta_hombre_fumador if cuenta_hombre_fumador != 0 else 0
print(f"La cantidad de hombres fumadores es: {cuenta_hombre_fumador}")
print(f"La cantidad de mujeres fumadoras es: {cuenta_mujer_fumador}")
print(f"La proporción de fumadoras por cada fumador es: {proporcion_mujer_x_hombre:.2f}")
separador()
separador()

print(' --- PROYECTO INTEGRADOR PARTE 4 --- ')

#1. Realiza un GET request para descargarlos y escribe la respuesta como un archivo de texto plano con extensión csv (no necesitas pandas para esto, sólo manipulación de archivos nativa de Python)
#2. Agrupa el código para esto en una función reutilizable que reciba como argumento la url.
def descarga_get_request (url, salida_archivo):
    response = requests.get(url)

    if response.status_code == 200:
        with open(salida_archivo, 'w', encoding='utf-8') as file:
            file.write(response.text)
        print(f"Los datos se han guardado en {salida_archivo}")
    else:
        print(f"No se pudo almacenar los datos. Código de respuesta: {response.status_code}")


url = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
salida_archivo = "heart_failure_clinical_records_dataset.csv"
descarga_get_request(url, salida_archivo)
separador()
separador()

print(' --- PROYECTO INTEGRADOR PARTE 5 --- ')

def procesar_dataframe(df: pd.DataFrame) -> None:
    # 1. Verificar que no existan valores faltantes
    if df.isnull().values.any():
        print("Hay valores faltantes. Se eliminarán las filas con valores faltantes")
        df = df.dropna()  

    # 2. Verificar que no existan filas repetidas
    if df.duplicated().any():
        print("Existen filas duplicadas en el DataFrame. Se eliminarán los duplicados")
        df = df.drop_duplicates()  

    # 3. Verificar si existen valores atípicos y eliminarlos 
    Q1 = df['age'].quantile(0.25)
    Q3 = df['age'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['age'] >= Q1 - 1.5 * IQR) & (df['age'] <= Q3 + 1.5 * IQR)]

    # 4. Crear una columna que categorice por edades
    bins = [0, 12, 19, 39, 59, float('inf')]
    labels = ['Niño', 'Adolescente', 'Adulto joven', 'Adulto', 'Adulto mayor']
    df['edad_categorizada'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # 5. Guardar el resultado como csv
    resultado_csv = "resultado_proyecto_int_PT5.csv"
    df.to_csv(resultado_csv, index=False)
    print(f"El resultado se ha guardado en {resultado_csv}")

# Carga el nuevo dataframe desde el archivo descargado
nuevo_dataframe = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Llama a la función con el nuevo dataframe
procesar_dataframe(nuevo_dataframe)
separador()
separador()
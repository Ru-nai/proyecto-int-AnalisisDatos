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
df = pd.DataFrame(data)
print(f"El Dataframe para este proyecto es: \n {df}")

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

promedio_edad_fallecido = round(np.mean(df_is_dead['age']), 2)
promedio_edad_nofallecido = round(np.mean(df_isnot_dead['age']), 2)
print(f"Promedio de edad de los fallecidos: {promedio_edad_fallecido} años")
print(f"Promedio de edad de los que NO fallecieron: {promedio_edad_nofallecido} años")
separador()
separador()

print(' --- PROYECTO INTEGRADOR PARTE 3 --- ')
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

cuenta_hombre_fumador = df[(df['is_smoker'] == True) & (df['is_male'] == True)].shape[0]
cuenta_mujer_fumador = df[(df['is_smoker'] == True) & (df['is_male'] == False)].shape[0]
print(f"La cantidad de hombres fumadores es: {cuenta_hombre_fumador}")
print(f"La cantidad de mujeres fumadoras es: {cuenta_mujer_fumador}")
separador()
separador()

print(' --- PROYECTO INTEGRADOR PARTE 4 --- ')

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

def procesar_dataframe(df):
    # 1. Verificar que no existan valores faltantes
    if df.isnull().values.any():
        print("Existen valores faltantes en el DataFrame. Se eliminarán las filas con valores faltantes.")
        df = df.dropna()  # Elimina filas con valores faltantes

    # 2. Verificar que no existan filas repetidas
    if df.duplicated().any():
        print("Existen filas duplicadas en el DataFrame. Se eliminarán las filas duplicadas.")
        df = df.drop_duplicates()  # Elimina filas duplicadas

    # 3. Verificar si existen valores atípicos y eliminarlos (puedes adaptar según tus necesidades)
    # Supongamos que quieres eliminar valores atípicos en la columna 'age' usando el rango intercuartílico
    Q1 = df['age'].quantile(0.25)
    Q3 = df['age'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['age'] >= Q1 - 1.5 * IQR) & (df['age'] <= Q3 + 1.5 * IQR)]

    # 4. Crear una columna que categorice por edades
    bins = [0, 12, 19, 39, 59, float('inf')]
    labels = ['Niño', 'Adolescente', 'Jóvenes adulto', 'Adulto', 'Adulto mayor']
    df['edad_categorizada'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # 5. Guardar el resultado como csv
    resultado_csv = "resultado_proyecto_integrador.csv"
    df.to_csv(resultado_csv, index=False)
    print(f"El resultado se ha guardado como {resultado_csv}")

# Carga el nuevo dataframe desde el archivo descargado
nuevo_dataframe = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Llama a la función con el nuevo dataframe
procesar_dataframe(nuevo_dataframe)
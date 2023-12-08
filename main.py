from datasets import load_dataset
import numpy as np
import pandas as pd #PT2

''' --- PROYECTO INTEGRADOR PARTE 1 --- '''
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

lista_edades = data['age']
arreglo_edades = np.array(lista_edades)
promedio_edades = round(np.mean(lista_edades), 2)

print(f"Promedio de edad de los participantes: {promedio_edades} años")

""" --- PROYECTO INTEGRADOR PARTE 2 --- """

#1. Convertir la estructura Dataset en un DataFrame de Pandas usando pd.DataFrame.
df = pd.DataFrame(data)
print(df)

#2. Separar el dataframe en dos diferentes, uno conteniendo las filas con personas que perecieron (is_dead=1) y otro con el complemento.
df_is_dead = df[df['is_dead'] == 1]
df_isnot_dead = df[df['is_dead'] == 0]

print(f"Dataframe de las personas que fallecieron: {df_is_dead}")
print(f"Dataframe de las personas que NO fallecieron: {df_isnot_dead}")

#3. Calcular los promedios de las edades de cada dataset e imprimir.
promedio_edad_fallecido = round(np.mean(df_is_dead['age']))
promedio_edad_nofallecido = round(np.mean(df_isnot_dead['age']))

print(f"Promedio de edad de los fallecidos: {promedio_edad_fallecido} años")
print(f"Promedio de edad de los que NO fallecieron: {promedio_edad_nofallecido} años")


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

df = pd.DataFrame(data)
print(df)

df_is_dead = df[df['is_dead'] == 1]
df_isnot_dead = df[df['is_dead'] == 0]

print(f"Dataframe de las personas que fallecieron: {df_is_dead}")
print(f"Dataframe de las personas que NO fallecieron: {df_isnot_dead}")

promedio_edad_fallecido = round(np.mean(df_is_dead['age']))
promedio_edad_nofallecido = round(np.mean(df_isnot_dead['age']))

print(f"Promedio de edad de los fallecidos: {promedio_edad_fallecido} años")
print(f"Promedio de edad de los que NO fallecieron: {promedio_edad_nofallecido} años")


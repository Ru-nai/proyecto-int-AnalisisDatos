--- INTRODUCCIÓN AL ANÁLISIS DE DATOS ---

El proyecto de este curso consiste en analizar el conjunto de datos introducido en esta sección, 
procesarlo, limpiarlo y finalmente ajustar modelos de machine learning 
para realizar predicciones sobre estos datos.


---- PROYECTO INTEGRADOR PARTE 1 ----

Para el desarrollo de esta etapa del proyecto necesitamos intalar la librería datasets de Huggingface

Vamos a trabajar con un dataset sobre fallo cardíaco

El dataset contiene registros médicos de 299 pacientes que padecieron insuficiencia cardíaca durante un período de seguimiento.

Las 13 características clínicas incluidas en el conjunto de datos son:

1. Edad: edad del paciente (años)
2. Anemia: disminución de glóbulos rojos o hemoglobina (booleano)
3. Presión arterial alta: si el paciente tiene hipertensión (booleano)
4. Creatinina fosfoquinasa (CPK): nivel de la enzima CPK en la sangre (mcg/L)
5. Diabetes: si el paciente tiene diabetes (booleano)
6. Fracción de eyección: porcentaje de sangre que sale del corazón en cada contracción (porcentaje)
7. Plaquetas: plaquetas en la sangre (kiloplaquetas/mL)
8. Sexo: mujer u hombre (binario)
9. Creatinina sérica: nivel de creatinina sérica en la sangre (mg/dL)
10. Sodio sérico: nivel de sodio sérico en la sangre (mEq/L)
11. Fumar: si el paciente fuma o no (booleano)
12. Tiempo: período de seguimiento (días)
13. [Objetivo] Evento de fallecimiento: si el paciente falleció durante el período de seguimiento (booleano)


---- PROYECTO INTEGRADOR PARTE 2 ---- 

Continuando con la anterior sección del proyecto integrador, ahora debes realizar lo siguiente:

1. Convertir la estructura Dataset en un DataFrame de Pandas usando pd.DataFrame.
2. Separar el dataframe en dos diferentes, uno conteniendo las filas con personas que perecieron (is_dead=1) y otro con el complemento.
3. Calcular los promedios de las edades de cada dataset e imprimir.


---- PROYECTO INTEGRADOR PARTE 3 ----

Continuando con el DataFrame con todos los datos de la anterior subsección, ahora debes:

1. Verificar que los tipos de datos son correctos en cada colúmna (por ejemplo que no existan colúmnas numéricas en formato de cadena).
2. Calcular la cantidad de hombres fumadores vs mujeres fumadoras (usando agregaciones en Pandas).
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


---- PROYECTO INTEGRADOR PARTE 4 ----

Imagina que no tuvieramos el acceso fácil de estos datos a través de la librería y tuvieramos que descargar los datos usando requests.

Los datos son accesibles en esta dirección: https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv

1. Realiza un GET request para descargarlos y escribe la respuesta como un archivo de texto plano con extensión csv (no necesitas pandas para esto, sólo manipulación de archivos nativa de Python)
2. Agrupa el código para esto en una función reutilizable que reciba como argumento la url.


---- PROYECTO INTEGRADOR PARTE 5 ----

Una vez cargado el csv mediante el request anterior, realiza lo siguiente:

1. Verificar que no existan valores faltantes
2. Verificar que no existan filas repetidas
3. Verificar si existen valores atípicos y eliminarlos
4. Crear una columna que categorice por edades
    - 0-12: Niño
    - 13-19: Adolescente
    - 20-39: Jóvenes adulto
    - 40-59: Adulto
    - 60-...: Adulto mayor
5. Guardar el resultado como csv

Encapsula toda la lógica anterior en una función que reciba un dataframe como entrada.


---- PROYECTO INTEGRADOR PARTE 6 ----

Imagina que los datos que procesaste en anteriores etapas del proyecto integrador se van actualizando cada cierto tiempo, (manteniendo el formato) y la url siempre va apuntando a la versión más actual, en este caso conviene tener automatizado el procesamiento en un script que pedas llamar y siempre te dé un csv limpio y listo para su análisis.

Tu tarea en esta etapa del proyecto consiste en crear un script (un archivo .py) que realice todas las operaciones vistas hasta ahora (desde el módulo de APIS) que al ejecutarse:

    - Descargue los datos desde una url
    - Convierta todo a un dataframe
    - Categorice en grupos
    - Exporte un csv resultante

La url no debe estar definida como una constante en el código, en su lugar usa argumentos por terminal (revisar este enlace: https://www.geeksforgeeks.org/how-to-use-sys-argv-in-python/) para enviarle la url al programa al ejecutarlo.


---- PROYECTO INTEGRADOR PARTE 7 ----

Una vez tenemos los datos exportados por nuestro script de ETL, podemos proceder a realizar gráficas de análisis. En esta etapa del proyecto usa matplotlib para:

1. Graficar la distribución de edades con un histograma
2. Graficar histogramas agrupado por hombre y mujer:
    - cantidad de anémicos
    - cantidad de diabéticos
    - cantidad de fumadores
    - cantidad de muertos

El segundo histograma debe verse así: https://media.ada-school.org/5fcd3ac12b22eab4d301d819/61345ed31a244b00166eb22c/figure_1-9757dc9d-1ae7-47b3-b30b-ba9aa6afcfd5.png

Tip:
    Para graficar barras lado a lado puedes usar el argumento align='edge' de la función plt.bar y definir un ancho positivo y otro negativo para evitar sobreponerlos.


---- PROYECTO INTEGRADOR PARTE 8 ----

Usando el mismo DataFrame, realiza una gráfica usando subplots, que contenga gráficas de torta que represente las distribuciones de:

    - Cantidad de anémicos
    - Cantidad de diabéticos
    - Cantidad de fumadores
    - Cantidad de muertos

La grafica debe verse similar a esta (no es necesario el mismo color)

https://media.ada-school.org/5fcd3ac12b22eab4d301d819/61345ed31a244b00166eb22c/figure_2-32e72543-83e1-4fe3-9a16-80976778b67e.png
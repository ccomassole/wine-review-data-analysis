
# coding: utf-8

# # PRACTICA2. Tipologia de Datos
# Preparación y analisis de un dataset
# ------------------------------------------------------

# Script de preparación y análisis del Dataset "Wine Review". El objetivo planteado es verificar si existen relaciones significativas entre entre el precio de un vino, su valoración y la descripción que del mismo realiza un sommelier experto. Para simplificar el análisis, se tomará como elemento de valoración de la descripción la longitud de la misma. De este modo el analisis se centrará exclusivamente en valores numericos.
# 
# La numeración dada a cada apartado se corresponde con la del documento explicativo.

# ## 2.6.1. Análisis de conjunto

# El proceso de análisis de conjunto identificará el número total de registros (muestras) del dataset, así como los atributos, sus tipos y las estadisticas generales de los datos (promedios, varianza, etc.).

# In[13]:


get_ipython().magic(u'matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import percentile
from matplotlib import pyplot
from scipy import stats
from scipy.stats import normaltest

# Cargamos los datos del fichero "winemag-data-130k-v2.csv" en un dataframe

path = "C:/Users/Usuario/CloudStation/02 Emprender/03 HolaParlem/10 Biblioteca/06 Master Data Science UOC/97 Entregas/06 Tipologia de Datos/PRACTICA2/Dataset"
file_name="winemag-data-130k-v2.csv"
data = pd.read_csv(path + "/" + file_name)

def info(df):
    # Mostramos el número de muestras del fichero
    print ("Número de registros: " + str(len(df)) + "\n")

    # Mostramos el número de muestras del fichero
    print ("Número de atributos: " + str(len(df.columns)) + "\n")

    # Mostramos los atributos junto con el tipo asignado en la extracción
    print "Analisis del tipado de los datos:"
    print(df.dtypes)
    print ("\n")
    
    # Valores únicos de cada categoría/atributo
    print "Valores únicos por categoría:"
    print(data.nunique())
    print ("\n")
    
    # Analisis estadistico básico
    print "Analisis estadistico básico:"
    print (df.describe())
    print ("\n")

info(data)
    
# Visualizamos algunos datos (filas i a j)
print(data[1:10])
print ("\n")

# Nombramos los atributos que no están identificados (ID)
data.rename(columns={'Unnamed: 0':'id'}, inplace=True)
print (data.columns)
print ("\n")


# Conclusiones:
# ---
# 1.- El número total de registros es de: 129.971
# 2.- El número de vinos diferentes es muy alto (prácticamente hay un registro por vino)
# 3.- 707 variedades de uva
# 4.- Vinos de hasta 43 paises diferentes
# 5.- Los tipos de las variables numericas asignadas por defecto son coherentes a su función. No se realizarán cambios en ellas.
# 6.- Se nombra el atributo (de origen vacio), cuya función es identificar cada registro como 'id'.
# 7.- Todas las puntuaciones de los vinos identificados son superiores a 80, que es consistente con la información proporcionada en la página origen de los datos.
# 8.- El precio se encuentra concentrado en una horquilla de hasta 50 €, observandose la existencia de valores muy por debajo y muy por encima de la media. En consecuencia el análisis de valores extremos debe ser considerado particularmente relevante en este caso.

# ## 2.6.4. Limpieza

# ### 2.6.4.1.- Selección de atributos
# A priori el 'taster_name', el 'taster_twitter_handle', 'region_1', 'region_2' podrian eliminarse del dataset para su simplificación, ya que son variables que no contemplamos dentro de este análisis y de esta forma aligeramos el tratamiento de datos. Hay otras variables que tampoco forman parte del análisis pero preferimos no eliminarlas puesto que consideramos que pueden ser utiles de cara a completar posibles valores faltantes. Por otra parte incorporamos la longitud del texto de la descripcion como variable adicional.

# In[14]:


# Separamos los registros que no son de interés del dataset, eliminandolas

del data['taster_name']
del data['taster_twitter_handle']
del data['region_1']
del data['region_2']
data['descrip_length'] =data.description.str.len()
data.head(5)


# ### 2.6.4.2.- Duplicados
# Buscamos si existen muestras duplicadas.

# In[15]:


# Creamos una fila que identifique si hay duplicados
data['esta_duplicado']= data.duplicated()
print("El número de muestras duplicadas es de %d.") %len(data[data['esta_duplicado']==True])
# Eliminamos la fila creada
del data['esta_duplicado']


# ### 2.6.4.3.- Análisis de registros vacíos o nulos

# En primer lugar identificamos el número de registros vacios para cada atributo:

# In[16]:


# Contamos los registros vacios y calculamos el porcentaje que representan
def vacios(df):
    muestras_vacias = df.isnull().sum()
    # identificamos su peso en cada caso:
    total_registros = np.product(df.shape)
    total_vacias = muestras_vacias.sum()
    return ((float(total_vacias)/float(total_registros)) * 100)


print ("El numero de muestras vacias representa el %f porciento del total.") %vacios(data)


# verificamos que los campos vacios encontrados tanto en 'country' como en 'province' corresponden a las mismas muestras,
#al observar que ambos campos tienen el mismo numero de registros vacios.
if (len(data.loc[data.title.isin(data[data.country.isnull() & data.province.isnull()].title)])==len(data[data.country.isnull()])):
    print('Las muestras de country vacias tambien tienen province vacio')
else:
    print('Las muestras de country vacias NO coinciden necesariamente con province vacio')

# verificamos si podemos completar la información de variedad faltante (dado que solo hay un registro) 
# a partir del texto de descripcion o del titulo del vino
pd.set_option('display.max_colwidth', -1)
print(data[data.variety.isnull()].title)
print(data[data.variety.isnull()].description)
pd.set_option('display.max_colwidth', 50)

                                                                            


# Conclusiones:
# ---
# 1.- Solo el 3,6% de los registros estan vacios
# 2.- El mayor número de muestras vacias corresponde a la característica 'designation' (viñedo)
# 3.- Todas las muestras tienen valoracion (puntos y descripción)
# 4.- No se dispone de precio para un total de 8996 registros
# 5.- Aquellos campos que no disponen de información del pais, tampoco disponen de información de provincia y el vino de dichos campos, identificado por su titulo es único y en consecuencia no puede ser completada esta información a partir de otros valores.
# 6.- Solo un registro no tiene información de variedad (tipo de uva) y no observamos que podamos completarlo con la información de otros registros.
# 
# Decisiones:
# ---
# 1.- Eliminamos todos aquellos registros con algun campo de interés nulo. Dado que 'designation' no va a formar parte de dicho estudio, para evitar reducir en exceso el dataset, lo completamos como 'Unknow' y así lo mantendremos posteriormente al eliminar el resto que son vacios.
# 

# In[17]:


# Etiquetamos como desconocidos los campos vacios del atributo designation para mantenerlos
data['designation'] = data.designation.replace(np.NaN, 'Unknown')
# Eliminamos los registros vacios distinguiendo los casos en que los atributos son numericos o cadenas de texto.
atributtes=list(data)
for column_name in atributtes:
    if ((column_name != 'id') | (column_name != 'points') | (column_name != 'price')):
        data = data[pd.notnull(data[column_name])]
    else:
        data.dropna(axis=0, subset=[column_name])
        
# Comprobamos que hemos eliminado los registros vacios
info(data)
print ("El numero de muestras vacias representa el %f por ciento del total.") %vacios(data)


# ### 2.6.4.4.- Outliers

# Procedemos a la identificación de valores extremos para los registros numericos de precio, valoracion y longitud del texto de descripcion. Lo hacemos mediante analisis visual y aplicando el test de D'Agostino's (se ha considerado este test puesto que el numero de registros disponibles es elevado)

# In[18]:


# Outliers: analisis del registro de precio

#comprobamos visualmente la normalidad
plt.figure(figsize=(16,6))
plt.subplot(1,1,1)
g = sns.countplot(x='price', data=data)
plt.show()

#comprobamos la normalidad mediante test (D’Agostino’s K^2 Test)
stat, p = normaltest(data['price'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpretamos el resultado
alpha = 0.05
if p > alpha:
    print('Las muestras parecen seguir una distribución normal')
else:
    print('Las muestras no siguen una distribución normal.')  


#identificamos visualmente los outliers
plt.figure(figsize=(16,6))
plt.subplot(1,1,1)
sns.boxplot(x=data['price'])
plt.show()


# identificar los outliers considerando que la muestra no es normal. (Metodo de intercuartiles)
q25, q75 = percentile(data['price'], 25), percentile(data['price'], 75)
iqr = q75 - q25
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
# calculamos los puntos de corte de los outliers
cut_off = iqr * 1.5
lower_price, upper_price = q25 - cut_off, q75 + cut_off
# identificamos los outliers
outliers = [x for x in data['price'] if x < lower_price or x > upper_price]
print('Outliers: %d' % len(outliers))


# In[19]:


# Outliers: analisis del registro de puntos

#comprobamos visualmente la normalidad
plt.figure(figsize=(16,6))
plt.subplot(1,1,1)
g = sns.countplot(x='points', data=data)
plt.show()

#comprobamos la normalidad mediante test (D’Agostino’s K^2 Test)
stat, p = normaltest(data['points'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Las muestras parecen seguir una distribución normal')
else:
    print('Las muestras no siguen una distribución normal.')  


#identificamos visualmente los outliers
plt.figure(figsize=(16,6))
plt.subplot(1,1,1)
sns.boxplot(x=data['points'])
plt.show()


# identificar los outliers considerando que la muestra no es normal. (Metodo de intercuartiles)
q25, q75 = percentile(data['points'], 25), percentile(data['points'], 75)
iqr = q75 - q25
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
# calculamos los puntos de corte de los outliers
cut_off = iqr * 1.5
lower_points, upper_points = q25 - cut_off, q75 + cut_off
# identificamos los outliers
outliers = [x for x in data['points'] if x < lower_points or x > upper_points]
print('Outliers: %d' % len(outliers))


# In[20]:


# Outliers: analisis del registro de longitud de la descripcion

#comprobamos visualmente la normalidad
plt.figure(figsize=(16,6))
plt.subplot(1,1,1)
g = sns.countplot(x='descrip_length', data=data)
plt.show()

#comprobamos la normalidad mediante test (D’Agostino’s K^2 Test)
stat, p = normaltest(data['descrip_length'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Las muestras parecen seguir una distribución normal')
else:
    print('Las muestras no siguen una distribución normal.')  


#identificamos visualmente los outliers
plt.figure(figsize=(16,6))
plt.subplot(1,1,1)
sns.boxplot(x=data['descrip_length'])
plt.show()


# identificar los outliers considerando que la muestra no es normal. (Metodo de intercuartiles)
q25, q75 = percentile(data['descrip_length'], 25), percentile(data['descrip_length'], 75)
iqr = q75 - q25
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
# calculamos los puntos de corte de los outliers
cut_off = iqr * 1.5
lower_desc, upper_desc = q25 - cut_off, q75 + cut_off
# identificamos los outliers
outliers = [x for x in data['descrip_length'] if x < lower_desc or x > upper_desc]
print('Outliers: %d' % len(outliers))


# Conclusiones
# -----
# 1.- Aunque visualmente tanto los atributos points como description_lenght parecen seguir una distribución normal, los test nos indican que no es así.
# 2.- Por ello hemos utilizado el método de intercuartiles para detectar los outliers.
# 3.- El número de outliers es muy elevado en el caso del precio (como ya se adelantaba) y también en el caso de las longitudes de la descripción.
# 
# Decisiones
# ------
# De cara a decidir que hacer con dichos valores extremos, consideramos que la existencia de extremos el precio es admisible, considerando que en el mercado existen vinos con precios prohibitivos, sin embargo creemos que tanto la puntuación como la longitud de la descripción que a priori son valores subjetivos pueden conducir, de incorporarlos a los datos, a errores en el análisis.

# In[21]:


# eliminamos los outliers de precio y longitud de descripcion obtenidos
data=data[((data['price']>= lower_price) & (data['price']<= upper_price))]
data=data[((data['descrip_length']>= lower_desc) & (data['descrip_length']<= upper_desc))]
info(data)


# ### 2.6.4.5.- Salvamos el fichero preparado
# 
# Procedemos a salvar el fichero con los datos preparados. El conjunto de registros final es de: 112184, con 11 atributos.
# 

# In[22]:


# Guardar un csv
data.to_csv("data_prepared.csv")


# ## 2.7.- Analisis

# Para proceder al análisis de la relacion existente entre el precio, la valoración y la longitud de la descripción realizaremos un análisis de correlación.
# Ya hemos verificado previamente que ninguna de las variables presenta un comportamiento normal. Es por ello que realizaremos el cálculo del coeficiente de correlación de las diferentes variables con respecto al precio utilizando el coeficiente de Spearman. 

# In[23]:


#carga de librerias estadísticas (necesarias para el calculo del coeficiente de correlacion)
from scipy.stats import spearmanr

# cargamos el fichero con los datos preparados
path = "C:/Users/Usuario/00 Practicas UOC"
file_name="data_prepared.csv"
data = pd.read_csv(path + "/" + file_name)

#Analisis gráfico previo
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
g = sns.regplot(x='points', y='price', data=data, x_jitter=True, fit_reg=False)
g.set_title("Distribucion: Puntos x Precio", fontsize=20)
g.set_xlabel("Puntos", fontsize= 15)
g.set_ylabel("Precio", fontsize= 15)
plt.subplot(1,2,2)
g = sns.regplot(x='descrip_length', y='price', data=data, x_jitter=True, fit_reg=False)
g.set_title("Distribucion: Longitud descipcion x Precio", fontsize=20)
g.set_xlabel("Longitud descipcion", fontsize= 15)
g.set_ylabel("Precio", fontsize= 15)
plt.show()

#Calculo del coeficiente de cortrelación de spearman
corr, p_value = spearmanr(data['price'], data['points'])
print corr

corr, p_value = spearmanr(data['price'], data['descrip_length'])
print corr



# Conclusiones
# ----
# Considerando que en el rango [-1,1] ambos valores son positivos, observamos que en ambos casos hay influencia entre las variables, si bien el atributo de valoracion es mas infuyente en el precio que la longitd de la descripción.

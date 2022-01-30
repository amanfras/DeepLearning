# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 01:29:54 2021

@author: Adela
"""



import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy  as np  
from sklearn.feature_selection import f_regression, mutual_info_regression


from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate

import cv2
#%% Nos quedamos con los datos de la ciudad de madrid ya que hemos decidido hacer un estudio centrado en esta ciudad
 
dataframe = pd.read_csv("D:/keepcoding/deep-learning/project/solucion/dataset/airbnb-listings.csv", sep=';')
dataframe = dataframe[dataframe['City']=='Madrid']

#%% Arreglar los códigos postales debido a que varios de ellos se han introducido erróneamenrte en el dataset y es preferible solucionarlo antes de hacer nada

dataframe['Zipcode'][dataframe['Zipcode'] == ""] = '0'
dataframe['Zipcode'][dataframe['Zipcode'] == "-"] = '0'
dataframe['Zipcode'][dataframe['Zipcode'] == '28'] = '0'

dataframe['Zipcode'][dataframe['Zipcode'] == 'Madrid 28004'] = '28004'

dataframe['Zipcode'][dataframe['Zipcode'] == '28002\n28002'] = '28002'
dataframe['Zipcode'][dataframe['Zipcode'] == '28051\n28051'] = '28051'

dataframe['Zipcode'][dataframe['Zipcode'] == '280013'] = '28013'
dataframe['Zipcode'][dataframe['Zipcode'] == '2015'] = '28015'
dataframe['Zipcode'][dataframe['Zipcode'] == '2815'] = '28015'
dataframe['Zipcode'][dataframe['Zipcode'] == '2805'] = '28005'
dataframe['Zipcode'][dataframe['Zipcode'] == '2804'] = '28004'

#%% Vamos a ver si hay alguna variable que nos interesa eliminar desde el principio por su tipo y que no nos aportan información

tipos = dataframe.dtypes

dataframe = dataframe.drop(['ID', 'Listing Url', 'Scrape ID', 'Last Scraped', 'Name', 'Summary', 'Space', 'Description', 'Experiences Offered', 'Neighborhood Overview', 'Notes', 'Transit', 'Access', 'Interaction', 'House Rules', 'Thumbnail Url', 'Medium Url', 'Picture Url', 'XL Picture Url', 'Host ID', 'Host URL', 'Host Name', 'Host Since', 'Host Location', 'Host About', 'Host Response Time', 'Host Response Rate', 'Host Acceptance Rate', 'Host Thumbnail Url', 'Host Picture Url', 'Host Total Listings Count', 'Host Verifications', 'Street', 'Host Neighbourhood', 'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed', 'City', 'State', 'Market', 'Smart Location', 'Country Code', 'Country', 'Weekly Price', 'Monthly Price', 'Calendar Updated', 'Has Availability', 'Availability 30', 'Availability 60', 'Availability 90', 'Availability 365', 'Calendar last Scraped', 'First Review', 'Last Review', 'License', 'Jurisdiction Names', 'Calculated host listings count', 'Geolocation', 'Features'], axis=1)

#De este modo se ha decidido eliminar todas las variables en las que se describe el airbnb a través de frases, las url, los ID que no aportan datos, los datos de localización de ciudad y país.. debido a que ya se ha preseleccionado la ciudad de Madrid y no tendría sentido emplear estos campos, los de precios ya que no tiene sentido emplearlos para predecirlo, etc.

#%% Dividimos en train y test

dataframe_train, dataframe_test = train_test_split(dataframe, test_size = 0.3, shuffle = False, random_state = 0)

# Codificamos las variables de tipo string

dataframe_train['Neighbourhood'].fillna('Other', inplace=True)
le1 = preprocessing.LabelEncoder()
le1.fit(dataframe_train['Neighbourhood'])
dataframe_train['Neighbourhood'] = le1.transform(dataframe_train['Neighbourhood'])

dataframe_train['Zipcode'].fillna(dataframe_train['Zipcode'].mode()[0], inplace=True)
dataframe_train['Zipcode'] = dataframe_train['Zipcode'].astype(int)

le2 = preprocessing.LabelEncoder()
le2.fit(dataframe_train['Property Type'])
dataframe_train['Property Type'] = le2.transform(dataframe_train['Property Type'])

le3 = preprocessing.LabelEncoder()
le3.fit(dataframe_train['Room Type'])
dataframe_train['Room Type'] = le3.transform(dataframe_train['Room Type'])

le4 = preprocessing.LabelEncoder()
le4.fit(dataframe_train['Bed Type'])
dataframe_train['Bed Type'] = le4.transform(dataframe_train['Bed Type'])

dataframe_train['Amenities'].fillna('', inplace=True)
dataframe_train['Amenities'] = dataframe_train['Amenities'].apply(lambda x: len(x.split(',')))

le5 = preprocessing.LabelEncoder()
le5.fit(dataframe_train['Cancellation Policy'])
dataframe_train['Cancellation Policy'] = le5.transform(dataframe_train['Cancellation Policy'])

tipos = dataframe_train.dtypes # Ya son todos int/float

# Vemos el número de nans por si nos interesa quitarnos más variables o rellenar

nans = dataframe_train.isna().sum() 

dataframe_train['Host Listings Count'].fillna(dataframe_train['Host Listings Count'].mode()[0], inplace=True)
dataframe_train['Bathrooms'].fillna(dataframe_train['Bathrooms'].mode()[0], inplace=True)
dataframe_train['Bedrooms'].fillna(dataframe_train['Bedrooms'].mode()[0], inplace=True)
dataframe_train['Beds'].fillna(dataframe_train['Beds'].mode()[0], inplace=True)
dataframe_train = dataframe_train.drop(['Square Feet'], axis=1) # Porcentaje muy alto de valores nan no nos interesa
dataframe_train['Price'].fillna(dataframe_train['Price'].mode()[0], inplace=True)
dataframe_train = dataframe_train.drop(['Security Deposit'], axis=1)
dataframe_train['Cleaning Fee'].fillna(0, inplace=True) # Porcentaje muy alto de valores nan no nos interesa
dataframe_train['Review Scores Rating'].fillna(dataframe_train['Review Scores Rating'].mode()[0], inplace=True)
dataframe_train['Review Scores Accuracy'].fillna(dataframe_train['Review Scores Accuracy'].mode()[0], inplace=True)
dataframe_train['Review Scores Cleanliness'].fillna(dataframe_train['Review Scores Cleanliness'].mode()[0], inplace=True)
dataframe_train['Review Scores Checkin'].fillna(dataframe_train['Review Scores Checkin'].mode()[0], inplace=True)
dataframe_train['Review Scores Communication'].fillna(dataframe_train['Review Scores Communication'].mode()[0], inplace=True)
dataframe_train['Review Scores Location'].fillna(dataframe_train['Review Scores Location'].mode()[0], inplace=True)
dataframe_train['Review Scores Value'].fillna(dataframe_train['Review Scores Value'].mode()[0], inplace=True)
dataframe_train['Reviews per Month'].fillna(dataframe_train['Reviews per Month'].mode()[0], inplace=True)


#%% Comenzamos el análisis exploratorio

descripcion = dataframe_train.describe().T

#Plots de las variables que tiene sentido mirar si hay outliers basándonos en la descripción anterior

dataframe_train['Host Listings Count'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Host Listings Count')
dataframe_train_2 = dataframe_train[dataframe_train['Host Listings Count'] < 210] # Quitamos outliers

dataframe_train_2['Latitude'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Latitude')

dataframe_train_2['Longitude'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Longitude')

dataframe_train_2['Accommodates'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Accommodates')
dataframe_train_2['Accommodates'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Accommodates'] < 13] # Quitamos outliers

dataframe_train_2['Bathrooms'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Bathrooms')
dataframe_train_2['Bathrooms'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Bathrooms'] < 6.1] # Quitamos outliers

dataframe_train_2['Bedrooms'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Bedrooms')
dataframe_train_2['Bedrooms'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Bedrooms'] < 6.1] # Quitamos outliers

dataframe_train_2['Beds'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Beds')
dataframe_train_2['Beds'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Beds'] < 11] # Quitamos outliers

dataframe_train_2['Guests Included'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Guests Included')
dataframe_train_2['Guests Included'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Guests Included'] < 10] # Quitamos outliers

dataframe_train_2['Extra People'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Extra People')
dataframe_train_2['Extra People'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Extra People'] < 50] # Quitamos outliers

dataframe_train_2['Price'].plot.hist(alpha=0.5, bins=25, grid = True)
plt.xlabel('Price')
dataframe_train_2['Price'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Price'] < 400] # Quitamos outliers

# Scatter plot de la variable objetivo definida y como variable dependiente y algunas de las variables explicativas como independientes, en el caso de algunas de las codificadas usaremos waterfront

dataframe_train_2.plot(kind = 'scatter',x='Host Listings Count',y = 'Price')
dataframe_train_2.boxplot(by='Room Type',column = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Accommodates',y = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Bathrooms',y = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Bedrooms',y = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Beds',y = 'Price')
dataframe_train_2.boxplot(by='Bed Type',column = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Amenities',y = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Guests Included',y = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Extra People',y = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Review Scores Rating',y = 'Price')
dataframe_train_2.boxplot(by='Cancellation Policy',column = 'Price')
dataframe_train_2.plot(kind = 'scatter',x='Minimum Nights',y = 'Price')
dataframe_train_2['Minimum Nights'].value_counts()
dataframe_train_2 = dataframe_train_2[dataframe_train_2['Minimum Nights'] < 50] # Quitamos outliers

print(f'Porcentaje de registros eliminados: {((dataframe_train.shape[0] - dataframe_train_2.shape[0])/dataframe_train.shape[0])*100}%')


# Vamos a ver si hay colinealidad 

corr = np.abs(dataframe_train_2.drop(['Price'], axis=1).corr())
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
            linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})

# Alta colinealidad entre beds y accommodates, eliminamos accommodates, vemos que hay relación entre las review scores, en concreto:
# Review Scores Rating = (Review Scores Accuracy + Review Scores Cleanliness + Review Scores Checkin + Review Scores Communication + Review Scores Location + Review Scores Value)*100/60
# Nos quedaremos con Review Scores Rating que es la combinación de las demás

dataframe_train_final = dataframe_train_2.drop(['Accommodates', 'Review Scores Accuracy', 'Review Scores Cleanliness', 'Review Scores Checkin', 'Review Scores Communication', 'Review Scores Location', 'Review Scores Value'], axis=1)
pd.plotting.scatter_matrix(dataframe_train_final, alpha=0.2, figsize=(20, 20), diagonal = 'kde')

# Vamos a ver la relación con la variable dependiente, dividimos en variable dependiente e independientes

Y_train = dataframe_train_final['Price']
X_train = dataframe_train_final.drop(['Price'], axis=1)

feature_names = X_train.columns
f_test, _ = f_regression(X_train, Y_train)
f_test /= np.max(f_test)
mi = mutual_info_regression(X_train, Y_train)
mi /= np.max(mi)

plt.figure(figsize=(20, 5))

plt.subplot(1,2,1)
plt.bar(range(X_train.shape[1]),f_test,  align="center")
plt.xticks(range(X_train.shape[1]),feature_names, rotation = 90)
plt.xlabel('features')
plt.ylabel('Ranking')
plt.title('$FTest$ score')

plt.subplot(1,2,2)
plt.bar(range(X_train.shape[1]),mi, align="center")
plt.xticks(range(X_train.shape[1]),feature_names, rotation = 90)
plt.xlabel('features')
plt.ylabel('Ranking')
plt.title('Mutual information score')

# Con estos resultados vamos a eliminar un par de variables más debido a su puntuación baja en el Ftest y mutual information score

X_train = X_train.drop(['Property Type', 'Bed Type', 'Maximum Nights'], axis=1)

#%% Preparamos para modelado, para ello vamos a realizar en el test los fillna y codificaciones que hicimos en el train

dataframe_test['Price'].fillna(dataframe_test['Price'].mode()[0], inplace=True)
dataframe_test = dataframe_test[dataframe_test['Price'] < 400]

dataframe_test = dataframe_test.drop(['Square Feet', 'Security Deposit', 'Property Type', 'Bed Type', 'Maximum Nights', 'Accommodates', 'Review Scores Accuracy', 'Review Scores Cleanliness', 'Review Scores Checkin', 'Review Scores Communication', 'Review Scores Location', 'Review Scores Value'], axis=1)

dataframe_test['Host Listings Count'].fillna(dataframe_test['Host Listings Count'].mode()[0], inplace=True)
dataframe_test = dataframe_test[dataframe_test['Host Listings Count'] < 210]

dataframe_test['Neighbourhood'].fillna('Other', inplace=True)
dataframe_test['Neighbourhood'] = le1.transform(dataframe_test['Neighbourhood'])

dataframe_test['Zipcode'].fillna(dataframe_test['Zipcode'].mode()[0], inplace=True)
dataframe_test['Zipcode'] = dataframe_test['Zipcode'].astype(int)

dataframe_test['Room Type'] = le3.transform(dataframe_test['Room Type'])

dataframe_test['Bathrooms'].fillna(dataframe_test['Bathrooms'].mode()[0], inplace=True)
dataframe_test = dataframe_test[dataframe_test['Bathrooms'] < 6.1]

dataframe_test['Bedrooms'].fillna(dataframe_test['Bedrooms'].mode()[0], inplace=True)
dataframe_test = dataframe_test[dataframe_test['Bedrooms'] < 6.1]

dataframe_test['Beds'].fillna(dataframe_test['Beds'].mode()[0], inplace=True)
dataframe_test = dataframe_test[dataframe_test['Beds'] < 11]

dataframe_test['Amenities'].fillna('', inplace=True)
dataframe_test['Amenities'] = dataframe_test['Amenities'].apply(lambda x: len(x.split(',')))

dataframe_test['Cleaning Fee'].fillna(0, inplace=True) 

dataframe_test = dataframe_test[dataframe_test['Guests Included'] < 10]

dataframe_test = dataframe_test[dataframe_test['Extra People'] < 50]

dataframe_test = dataframe_test[dataframe_test['Minimum Nights'] < 50]

dataframe_test['Review Scores Rating'].fillna(dataframe_test['Review Scores Rating'].mode()[0], inplace=True)

dataframe_test['Cancellation Policy'] = le5.transform(dataframe_test['Cancellation Policy'])

dataframe_test['Reviews per Month'].fillna(dataframe_test['Reviews per Month'].mode()[0], inplace=True)


#%% Buscamos el precio maximo del dataframe para escalar los precios en el rango 0-1 y dividimos en variable dependiente e independientes
maxPrice = dataframe_train_final["Price"].max()
Y_train  = dataframe_train_final["Price"] / maxPrice
X_train = dataframe_train_final.drop(['Price'],  axis=1)
Y_test = dataframe_test["Price"] / maxPrice
X_test = dataframe_test.drop(['Price'],  axis=1)

#%% Transformamos los datos mediante un escalado min-max
continuous = ["Zipcode","Neighbourhood","Room Type", "Bathrooms", "Bedrooms","Beds","Amenities","Cleaning Fee","Guests Included","Extra People","Minimum Nights","Number of Reviews","Review Scores Rating","Cancellation Policy","Reviews per Month"]
# performin min-max scaling each continuous feature column to
# the range [0, 1]
cs = MinMaxScaler()
X_train = cs.fit_transform(X_train[continuous])
X_test = cs.transform(X_test[continuous])

#%% Creamos el modelo 
model = Sequential()
model.add(Dense(8, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="linear"))

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

#%% Entrenamos el modelo
print("Entrenando el modelo...")
model.fit(x=X_train, y=Y_train, 
validation_data=(X_train, Y_train),
epochs=20, batch_size=8)

#%% Predecimos el precio
print("Prediciendo el precio...")
preds = model.predict(X_test)
diff = preds.flatten() - Y_test
percentDiff = (diff / Y_test) * 100
absPercentDiff = np.abs(percentDiff)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
print("Precio de casas: {}, std precio casas: {}€".format(dataframe["Price"].mean(),dataframe["Price"].std()))
print(" media: {:.2f}%, std: {:.2f}%".format(mean, std))

#%% CNN

#%% Cargamos las imagenes
dataframe=pd.concat([dataframe_train_final,dataframe_test])
dataframe = dataframe.drop(['Host Listings Count','Latitude','Longitude','Property Type','Bed Type','Maximum Nights'], axis=1)
images = []

for i in dataframe.index:
    image = cv2.imread("D:/keepcoding/deep-learning/project/solucion/images/image{}.jpg".format(str(i)))
    image = cv2.resize(image, (64, 64))
    images.append(image)
    
    
images = np.array(images)
images = images / 255.0

(X_train_att_photo, X_test_att_photo, X_train_img_photo, X_test_img_photo) = train_test_split(dataframe, images, test_size=0.3001, shuffle = False, random_state = 0)

#%% Buscamos el precio maximo del dataframe para escalar los precios en el rango 0-1 y dividimos en variable dependiente e independientes
maxPrice = X_train_att_photo["Price"].max()
Y_train_photos = X_train_att_photo["Price"] / maxPrice
X_train_att_photo = X_train_att_photo.drop(['Price'],  axis=1)
Y_test_photos = X_test_att_photo["Price"] / maxPrice
X_test_att_photo = X_test_att_photo.drop(['Price'],  axis=1)


#%% Definimos la entrada del modelo
inputs = Input(shape=(64, 64, 3))

for (i, f) in enumerate((16, 32, 64)):
	if i == 0:
		x = inputs
	x = Conv2D(f, (3, 3), padding="same")(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=-1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(16)(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = Dropout(0.5)(x)
x = Dense(4)(x)
x = Activation("relu")(x)
x = Dense(1, activation="linear")(x)

#%% Construimos el modelo
model_photos = Model(inputs, x)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model_photos.compile(loss="mean_absolute_percentage_error", optimizer=opt)

#%% Entrenamos el modelo
print("Entrenando el modelo...")
model_photos.fit(x=X_train_img_photo, y=Y_train_photos, 
    validation_data=(X_test_img_photo, Y_test_photos),
    epochs=2, batch_size=8)

#%% Predecimos el precio
print("Prediciendo el precio...")
preds = model_photos.predict(X_test_img_photo)
diff = preds.flatten() - Y_test_photos
percentDiff = (diff / Y_test_photos) * 100
absPercentDiff = np.abs(percentDiff)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
print("Precio de casas: {}, std precio casas: {}€".format(dataframe["Price"].mean(),dataframe["Price"].std()))
print(" media: {:.2f}%, std: {:.2f}%".format(mean, std))



#%% Combinamos las salidas de ambos modelos
combinedInput = concatenate([model.output, model_photos.output])
#%% Creamos las layers
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)
#%% Creamos el modelo final con la entrada de ambos modelos vistos anteriormente
model_final = Model(inputs=[model.input, model_photos.input], outputs=x)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model_final.compile(loss="mean_absolute_percentage_error", optimizer=opt)

#%% Entrenamos el modelo
print("Entrenando el modelo...")
model_final.fit(
	x=[X_train_att_photo, X_train_img_photo], y=Y_train,
	validation_data=([X_test_att_photo, X_test_img_photo], Y_test),
	epochs=20, batch_size=8)

#%% Predecimos el precio
print("Prediciendo precios...")
preds = model_final.predict([X_test_att_photo, X_test_img_photo])
diff = preds.flatten() - Y_test_photos
percentDiff = (diff / Y_test_photos) * 100
absPercentDiff = np.abs(percentDiff)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
print("Precio de casas: {}, std precio casas: {}€".format(dataframe["Price"].mean(),dataframe["Price"].std()))
print(" media: {:.2f}%, std: {:.2f}%".format(mean, std))
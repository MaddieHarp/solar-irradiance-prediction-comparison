#Harp 6/23/2025
#formatting and testing data of solar irradence for mayetta KS using Linear Regression model from Mohamad Fares El Hajj Chehade
#github from this model: https://github.com/MFHChehade/Solar-Irradiance-Forecasting-using-ANNs-from-Scratch/blob/main/Models/NeuralNetwork.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from LinearRegression import LR

#setting up data taken from csv file on solar irradance in Mayetta KS(pulls levels and weather factors)
#skips first 13 because of how NASA Power Project Sets up their csv files when exporting
df = pd.read_csv("data/DailySolarData2000to2024.csv", skiprows=13)
features = ['T2M', 'RH2M', 'WS10M', 'PRECTOTCORR']
#taking the independent vars like temp, humidity, wind speed, precipitation
X = df[features].values
#taking dependant, the solar irradance for that day
y = df[['ALLSKY_SFC_SW_DWN']].values

#with different information being looked at with different scales, must scale values
#using sklearn to scale the values given
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#average test/training set is 80% train 20% test, splitting the given data
split_index = int(0.8 * len(X_scaled))
#creates test and train x
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
#creates test and train y
y_train, y_test = y[:split_index], y[split_index:]

#using the linear regression model and data given, fitting the model with data
model = LR(optimizer="Batch", learning_rate=0.01, iterations=50)
model.fit(X_train, y_train, X_test, y_test)

#plots loss for given data/algorithm
model.plot_loss(optimizer="Batch")
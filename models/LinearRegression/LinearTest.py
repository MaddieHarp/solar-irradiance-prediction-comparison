#Harp 6/23/2025
#formatting and testing data of solar irradence for mayetta KS using Linear Regression model from Mohamad Fares El Hajj Chehade
#github from this model: https://github.com/MFHChehade/Solar-Irradiance-Forecasting-using-ANNs-from-Scratch/blob/main/Models/NeuralNetwork.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
#for github model: from LinearRegression import LR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#setting up data taken from csv file on solar irradance in Mayetta KS(pulls levels and weather factors)
#skips first 14 because of how NASA Power Project Sets up their csv files when exporting

df = pd.read_csv("data/DailySolarData2000to2004pt2.csv", skiprows=14)
print(df.columns.tolist())
#pd.to_datetime expects year month and day, changing names to fit to_datetime
df.rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}, inplace=True)

# Create datetime and seasonality features
df["DATE"] = pd.to_datetime(df[['year', 'month', 'day']])
df["DOY"] = df["DATE"].dt.dayofyear
df["DOY_sin"] = np.sin(2 * np.pi * df["DOY"] / 365)
df["DOY_cos"] = np.cos(2 * np.pi * df["DOY"] / 365)

df["clearness_index"] = df["ALLSKY_SFC_SW_DWN"] / df["CLRSKY_SFC_SW_DWN"]

#creating list of features or independent varaibles/X vars
features = ['T2M', 'RH2M', 'WS10M', 'PRECTOTCORR', 'DOY_sin', 'DOY_cos', 'clearness_index']
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

#checking if data was scaled properly
print("Mean of each feature (should be close to 0):")
print(np.mean(X_scaled, axis=0))

print("\nStandard deviation of each feature (should be close to 1):")
print(np.std(X_scaled, axis=0))

#using sklearn linear regression model
lr = LinearRegression()

#training model with train data set
lr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = lr.predict(X_test)

#the coefficients, this shows feature importance
print("Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)

#mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

#coefficient of determination: 1 is perfect prediction
print("Coefficient of determination (R²): %.2f" % r2_score(y_test, y_pred))

# Plot actual vs predicted
plt.plot(y_test[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.legend()
plt.title("Linear Regression Predictions vs Actual")
plt.show()



##CODE TO RUN LINEAR REGRESSION MODEL FROM https://github.com/MFHChehade/Solar-Irradiance-Forecasting-using-ANNs-from-Scratch/blob/main/Models/NeuralNetwork.py###

# #using the linear regression model and data given, fitting the model with data
# model = LR(optimizer="Batch", learning_rate=0.01, iterations=50)
# model.fit(X_train, y_train, X_test, y_test)

# #plots loss for given data/algorithm
# model.plot_loss(optimizer="Batch")

# # Predict using the trained linear regression model
# predictions = model.predict(X_test)

# # Plot predicted vs actual for a subset (e.g., 100 days)
# plt.figure(figsize=(12, 6))
# plt.plot(y_test[:100], label="Actual", marker='o')
# plt.plot(predictions[:100], label="Predicted", marker='x')
# plt.title("Linear Regression: Predicted vs Actual GHI")
# plt.xlabel("Sample (Day Index)")
# plt.ylabel("Solar Irradiance (kWh/m²)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
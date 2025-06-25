#Harp 6/23/2025
#formatting and testing data of solar irradence for mayetta KS using Linear Regression model from Mohamad Fares El Hajj Chehade
import pandas as pd
from sklearn.preprocessing import StandardScaler
#for github model: from NeuralNetwork import Net
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

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

# Scale y
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Split as before
split_index = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]

# Check mean and standard deviation of each feature in X_scaled
print("Mean of each feature (should be close to 0):")
print(np.mean(X_scaled, axis=0))

print("\nStandard deviation of each feature (should be close to 1):")
print(np.std(X_scaled, axis=0))

#creating ANN model from sklearns model
mlp = MLPRegressor(hidden_layer_sizes=(64, 48, 32), max_iter=500, learning_rate_init=0.01)
mlp.fit(X_train, y_train.ravel())  # .ravel() for MLPRegressor

predictions = mlp.predict(X_test)
predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))
y_test_unscaled = scaler_y.inverse_transform(y_test)

#plotting data
plt.plot(y_test_unscaled[:100], label="Actual")
plt.plot(predictions[:100], label="Predicted")
plt.legend()
plt.title("MLPRegressor Predictions vs Actual GHI")
plt.show()

print("Actual y_test range:", y_test_unscaled.min(), "to", y_test_unscaled.max())
print("Predicted range:", predictions.min(), "to", predictions.max())

##CODE TO RUN LINEAR REGRESSION MODEL FROM https://github.com/MFHChehade/Solar-Irradiance-Forecasting-using-ANNs-from-Scratch/blob/main/Models/NeuralNetwork.py###

# # Create ANN model
# model = Net(
#     layers=[X_train.shape[1], 64, 48, 32, 1],  # Input layer size = number of features (4)
#     learning_rate=0.01,
#     iterations=50
# )

# model.fit(X_train, y_train, X_test, y_test, optimizer="Batch")

# #plots loss for given data/algorithm
# model.plot_loss()

# predictions_scaled = model.predict(X_test)
# # Inverse-transform to get predictions in original units
# predictions = scaler_y.inverse_transform(predictions_scaled)
# y_test_unscaled = scaler_y.inverse_transform(y_test)

# plt.plot(y_test_unscaled[:100], label="Actual")
# plt.plot(predictions[:100], label="Predicted")
# plt.legend()
# plt.title("ANN Predictions vs Actual GHI")
# plt.show()

# print("Actual y_test range:", y_test_unscaled.min(), "to", y_test_unscaled.max())
# print("Predicted range:", predictions.min(), "to", predictions.max())

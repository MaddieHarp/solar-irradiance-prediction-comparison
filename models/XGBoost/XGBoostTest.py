#Harp 6/23/2025

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
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

# X_train, X_test, y_train, y_test are already prepared and scaled

# Create XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

# Train model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R² score: %.2f" % r2_score(y_test, y_pred))

# Plot actual vs predicted
plt.plot(y_test[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.legend()
plt.title("XGBoost Predictions vs Actual")
plt.show()
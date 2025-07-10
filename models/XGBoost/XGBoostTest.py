#Harp 6/23/2025
#formatting and testing data of solar irradence for mayetta KS using XGBoost
#implementation was from chen's XGBoost library
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
#chen's xgboost library
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.model_selection import learning_curve
import time

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

#shows how much memory the model takes up during training
#from memory_profiler import profile
# Train model
# @profile
# def train_model():
#     xgb_model.fit(X_train, y_train)
# train_model()

#displays time training took
start_time = time.time()
xgb_model.fit(X_train, y_train)
end_time = time.time()
#prints time taken
print(f"Training time: {end_time - start_time:.4f} seconds")

# Make predictions
y_pred = xgb_model.predict(X_test)

#calculates all comparsion metrics for model
# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Calculate RMSE (square root of MSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Calculate R² (coefficient of determination)
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2:.4f}")

# Plot actual vs predicted
plt.plot(y_test[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.legend()
plt.title("XGBoost Predictions vs Actual")
plt.xlabel("Sample Index")  
plt.ylabel("Solar Irradiance (kWh/m²/day)")  
#display graph
plt.show()

#this portion is to create and display learning curve graph
# Define RMSE scorer for learning_curve
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Build learning curve
train_sizes, train_scores, test_scores = learning_curve(
    xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1),
    X_scaled,
    y.ravel(),
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring=rmse_scorer,
    cv=5,
    shuffle=True,
    random_state=42
)

# Convert negative MSE to RMSE
train_rmse = np.sqrt(-train_scores.mean(axis=1))
test_rmse = np.sqrt(-test_scores.mean(axis=1))

# Plot learning curve
plt.plot(train_sizes, train_rmse, label="Training RMSE")
plt.plot(train_sizes, test_rmse, label="Validation RMSE")
plt.xlabel("Training Set Size")
plt.ylabel("RMSE (kWh/m²/day)")
plt.title("Learning Curve for XGBoost")
plt.legend()
plt.grid(True)
#display
plt.show()
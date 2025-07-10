#Harp 6/23/2025
#formatting and testing data of solar irradence for mayetta KS
#this implementation was off of Scikit Learn's MPLRegressor Library
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
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
#captures seasonality, ex: dec 31st and Jan 1st far apart according to year but not seasonally, fixes this
df["DOY_sin"] = np.sin(2 * np.pi * df["DOY"] / 365)
df["DOY_cos"] = np.cos(2 * np.pi * df["DOY"] / 365)

#clearness index, cloud coverage
df["clearness_index"] = df["ALLSKY_SFC_SW_DWN"] / df["CLRSKY_SFC_SW_DWN"]

#creating list of features or independent varaibles/X vars
features = ['T2M', 'RH2M', 'WS10M', 'PRECTOTCORR', 'DOY_sin', 'DOY_cos', 'clearness_index']

#taking the independent vars like temp, humidity, wind speed, precipitation
X = df[features].values
#taking dependant, the solar irradance for that day
y = df[['ALLSKY_SFC_SW_DWN']].values

#with different information being looked at with different scales, must scale values
#using sklearn to scale the values given with standarization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Scale y
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

#split data in 80-/20 split for training and testing
split_index = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]

#creating MLP(ANN) model from sklearns model
mlp = MLPRegressor(hidden_layer_sizes=(64, 48, 32), max_iter=500, learning_rate_init=0.01)

#This shows how much memory is used by the model
# from memory_profiler import profile
# Train model
# @profile
# def train_model():
#     mlp.fit(X_train, y_train.ravel()) 
# train_model()

#keeps track of how much training time is used
start_time = time.time()
#fits/trains the model with training data
mlp.fit(X_train, y_train.ravel())
end_time = time.time()
#prints training time for model
print(f"Training time: {end_time - start_time:.4f} seconds")

#model predicting future data
predictions = mlp.predict(X_test)
predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))
y_test_unscaled = scaler_y.inverse_transform(y_test)

#calculating all the comparasion metrics:

# Calculate MAE
mae = mean_absolute_error(y_test_unscaled, predictions)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Calculate MSE
mse = mean_squared_error(y_test_unscaled, predictions)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Calculate RMSE (square root of MSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Calculate R² (coefficient of determination)
r2 = r2_score(y_test_unscaled, predictions)
print(f"R-squared (R²): {r2:.4f}")

#plotting actul vs predicted data for model
plt.plot(y_test_unscaled[:100], label="Actual")
plt.plot(predictions[:100], label="Predicted")
plt.legend()
plt.title("MLPRegressor Predictions vs Actual GHI")
plt.xlabel("Sample Index") 
plt.ylabel("Solar Irradiance (kWh/m²/day)")  
#displays graph
plt.show()

#Define RMSE scorer (returns negative MSE, so we flip sign later)
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

#new model generated for learning curve graph
#Create learning curve for ANN
train_sizes, train_scores, test_scores = learning_curve(
    MLPRegressor(hidden_layer_sizes=(64, 48, 32), max_iter=500, learning_rate_init=0.01),
    X_scaled,
    y_scaled.ravel(),
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
plt.title("Learning Curve for ANN (MLPRegressor)")
plt.legend()
plt.grid(True)
#show learning curve
plt.show()
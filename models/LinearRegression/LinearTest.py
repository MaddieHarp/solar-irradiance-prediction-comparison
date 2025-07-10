#Harp 6/23/2025
#formatting and testing data of solar irradence for mayetta KS using Linear Regression model 
#implementation was friom sklearn's Linear regression library
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
#using sklearn to scale the values given
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#average test/training set is 80% train 20% test, splitting the given data
split_index = int(0.8 * len(X_scaled))
#creates test and train x
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
#creates test and train y
y_train, y_test = y[:split_index], y[split_index:]

#using sklearn linear regression model
lr = LinearRegression()

#This shows how much memory is used by the model
#from memory_profiler import profile
# Train model
# @profile
# def train_model():
#     lr.fit(X_train, y_train)
# train_model()

#training model with train data set
start_time = time.time()
lr.fit(X_train, y_train)
end_time = time.time()
#prints training time
print(f"Training time: {end_time - start_time:.4f} seconds")


# Make predictions using the testing set
y_pred = lr.predict(X_test)

#calculates all comparasion metrics for model
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
plt.title("Linear Regression Predictions vs Actual")
plt.xlabel("Sample Index") 
plt.ylabel("Solar Irradiance (kWh/m²/day)")
#displays plot
plt.show()


#this portion is for learning curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

# Assume you have X and y already defined
train_sizes, train_scores, test_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5,
    scoring='neg_mean_squared_error'
)

# Convert negative MSE to positive
train_scores = -train_scores
test_scores = -test_scores

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training MSE')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Validation MSE')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Learning Curve for Linear Regression')
plt.legend()
#displays learning curve
plt.show()
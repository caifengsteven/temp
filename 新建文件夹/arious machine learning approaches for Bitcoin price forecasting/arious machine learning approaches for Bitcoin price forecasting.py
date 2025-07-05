import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic Bitcoin price data (simulating 5 years of daily data)
days = 1825  # 5 years
dates = pd.date_range(start='2018-01-01', periods=days, freq='D')

# Create a trend component (overall upward with some volatility)
trend = np.linspace(3000, 60000, days) + np.random.normal(0, 5000, days).cumsum()

# Add seasonality (weekly and monthly patterns)
weekly_seasonality = 500 * np.sin(np.linspace(0, 2 * np.pi * (days/7), days))
monthly_seasonality = 2000 * np.sin(np.linspace(0, 2 * np.pi * (days/30), days))

# Add some randomness
noise = np.random.normal(0, 1000, days)

# Combine components
close_prices = trend + weekly_seasonality + monthly_seasonality + noise
close_prices = np.maximum(100, close_prices)  # Ensure no negative prices

# Generate other price features based on close price
open_prices = close_prices * np.random.normal(0.99, 0.01, days)
high_prices = close_prices * np.random.normal(1.05, 0.02, days)
low_prices = close_prices * np.random.normal(0.95, 0.02, days)

# Generate trading volume (correlated with price volatility)
volume = np.abs(np.diff(np.append(0, close_prices))) * np.random.normal(5000, 1000, days)

# Generate market cap (price * circulating supply)
circulating_supply = np.linspace(17000000, 19000000, days)  # Simulating increasing supply
market_cap = close_prices * circulating_supply

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices,
    'volume': volume,
    'market_cap': market_cap
})

# Display first few rows
print("Synthetic Bitcoin price data:")
print(df.head())

# Plot the closing prices
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['close'])
plt.title('Synthetic Bitcoin Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Data Preprocessing
# Create rolling features as mentioned in the paper
df['rolling_mean_7'] = df['close'].rolling(window=7).mean()
df['rolling_std_7'] = df['close'].rolling(window=7).std()
df['rolling_mean_30'] = df['close'].rolling(window=30).mean()
df['rolling_std_30'] = df['close'].rolling(window=30).std()

# Drop NaN values
df.dropna(inplace=True)

# Decomposition
result = seasonal_decompose(df['close'], model='additive', period=30)
plt.figure(figsize=(14, 10))

plt.subplot(411)
plt.plot(result.observed)
plt.title('Observed')

plt.subplot(412)
plt.plot(result.trend)
plt.title('Trend')

plt.subplot(413)
plt.plot(result.seasonal)
plt.title('Seasonal')

plt.subplot(414)
plt.plot(result.resid)
plt.title('Residual')

plt.tight_layout()
plt.show()

# Feature engineering
df['lag_1'] = df['close'].shift(1)
df['lag_2'] = df['close'].shift(2)
df['lag_3'] = df['close'].shift(3)
df['lag_5'] = df['close'].shift(5)
df['lag_7'] = df['close'].shift(7)
df['lag_14'] = df['close'].shift(14)
df['lag_30'] = df['close'].shift(30)

# Price momentum features
df['price_change_1'] = df['close'].pct_change(1)
df['price_change_7'] = df['close'].pct_change(7)
df['price_change_30'] = df['close'].pct_change(30)

# Volume features
df['volume_change_1'] = df['volume'].pct_change(1)
df['volume_7_avg'] = df['volume'].rolling(7).mean()
df['volume_30_avg'] = df['volume'].rolling(30).mean()

# Volatility features
df['volatility_7'] = df['price_change_1'].rolling(7).std()
df['volatility_30'] = df['price_change_1'].rolling(30).std()

# Drop NaN values after creating features
df.dropna(inplace=True)

# Prepare data for modeling
X = df.drop(['date', 'close'], axis=1)
y = df['close']

# Normalize features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Split data into train and test sets (80/20 as mentioned in the paper)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)
y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Implement the models mentioned in the paper
# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred_scaled = lr_model.predict(X_test)
lr_pred = scaler_y.inverse_transform(lr_pred_scaled.reshape(-1, 1)).flatten()
lr_mae = mean_absolute_error(y_test_original, lr_pred)
lr_score = r2_score(y_test_original, lr_pred)

# 2. Lasso Regression
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)
lasso_pred_scaled = lasso_model.predict(X_test)
lasso_pred = scaler_y.inverse_transform(lasso_pred_scaled.reshape(-1, 1)).flatten()
lasso_mae = mean_absolute_error(y_test_original, lasso_pred)
lasso_score = r2_score(y_test_original, lasso_pred)

# 3. Decision Tree
dt_model = DecisionTreeRegressor(max_depth=15, min_samples_leaf=10, min_impurity_decrease=0.5)
dt_model.fit(X_train, y_train)
dt_pred_scaled = dt_model.predict(X_test)
dt_pred = scaler_y.inverse_transform(dt_pred_scaled.reshape(-1, 1)).flatten()
dt_mae = mean_absolute_error(y_test_original, dt_pred)
dt_score = r2_score(y_test_original, dt_pred)

# 4. LSTM (reshape data for LSTM)
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[1])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_history = lstm_model.fit(
    X_train_lstm, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2,
    verbose=0
)

lstm_pred_scaled = lstm_model.predict(X_test_lstm).flatten()
lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
lstm_mae = mean_absolute_error(y_test_original, lstm_pred)
lstm_score = r2_score(y_test_original, lstm_pred)

# Print results
print("\nModel Evaluation Results:")
print(f"Linear Regression - MAE: {lr_mae:.2f}, Score: {lr_score:.4f}")
print(f"Lasso Regression - MAE: {lasso_mae:.2f}, Score: {lasso_score:.4f}")
print(f"Decision Tree - MAE: {dt_mae:.2f}, Score: {dt_score:.4f}")
print(f"LSTM - MAE: {lstm_mae:.2f}, Score: {lstm_score:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(14, 7))
test_dates = df.iloc[-len(y_test):]['date']

plt.plot(test_dates, y_test_original, label='Actual', color='black', linewidth=2)
plt.plot(test_dates, lr_pred, label='Linear Regression', linestyle='--')
plt.plot(test_dates, lasso_pred, label='Lasso Regression', linestyle='--')
plt.plot(test_dates, dt_pred, label='Decision Tree', linestyle='--')
plt.plot(test_dates, lstm_pred, label='LSTM', linestyle='--')

plt.title('Bitcoin Price Prediction: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a summary of feature importance for Lasso Regression
feature_names = X.columns
lasso_coefs = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso_model.coef_
})
lasso_coefs = lasso_coefs.sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(lasso_coefs['Feature'][:15], lasso_coefs['Coefficient'][:15])
plt.title('Top 15 Features by Importance (Lasso Regression)')
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Triple Exponential Smoothing (Holt-Winters) implementation
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Use a subset of data for this example
close_series = df['close'][-365:]  # Last year
train_data = close_series[:-30]  # Use all but last 30 days for training
test_data = close_series[-30:]   # Last 30 days for testing

# Fit Holt-Winters model
hw_model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='add',
    seasonal_periods=30  # Monthly seasonality
).fit()

# Make predictions
hw_forecast = hw_model.forecast(30)
hw_mae = mean_absolute_error(test_data, hw_forecast)

# Plot Holt-Winters results
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data, label='Training Data')
plt.plot(test_data.index, test_data, label='Actual Test Data')
plt.plot(test_data.index, hw_forecast, label='Holt-Winters Forecast')
plt.title('Bitcoin Price Forecast using Holt-Winters Method')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nHolt-Winters Method - MAE: {hw_mae:.2f}")

# Compare all models in a bar chart
methods = ['Linear Regression', 'Lasso Regression', 'Decision Tree', 'LSTM', 'Holt-Winters']
mae_values = [lr_mae, lasso_mae, dt_mae, lstm_mae, hw_mae]
score_values = [lr_score, lasso_score, dt_score, lstm_score, np.nan]  # No R² for Holt-Winters

plt.figure(figsize=(12, 6))
plt.bar(methods, mae_values)
plt.title('Mean Absolute Error Comparison Across Models')
plt.xlabel('Model')
plt.ylabel('MAE (USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a summary table
results_df = pd.DataFrame({
    'Model': methods,
    'MAE': mae_values,
    'R² Score': score_values
})
print("\nSummary of Model Performance:")
print(results_df)
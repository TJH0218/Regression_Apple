import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 讀取CSV資料
df = pd.read_csv('processed_apple_quality.csv')

# 選擇特徵和目標變數
X = df[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Acidity']]
y = df['Ripeness']

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立Random Forest Regression模型，設定一些超參數以避免overfitting
rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=5, random_state=42)
rf_model.fit(X_train, y_train)
y_rf_train_pred = rf_model.predict(X_train)
y_rf_test_pred = rf_model.predict(X_test)
rf_mse_train = mean_squared_error(y_train, y_rf_train_pred)
rf_r2_train = r2_score(y_train, y_rf_train_pred)
rf_mse_test = mean_squared_error(y_test, y_rf_test_pred)
rf_r2_test = r2_score(y_test, y_rf_test_pred)
print("\nRandom Forest Regression:")
print(f'Training MSE: {rf_mse_train:.3f}')
print(f'Test MSE: {rf_mse_test:.3f}')
print(f'Training R-squared: {rf_r2_train:.3f}')
print(f'Test R-squared: {rf_r2_test:.3f}')

# 繪製殘差圖
plt.figure(figsize=(8, 6))
plt.scatter(y_rf_train_pred, y_rf_train_pred - y_train, c='steelblue', edgecolor='white', marker='o', label='Training data')
plt.scatter(y_rf_test_pred, y_rf_test_pred - y_test, c='limegreen', edgecolor='white', marker='s', label='Test data')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Random Forest Regression')
plt.hlines(y=0, xmin=min(y_rf_train_pred), xmax=max(y_rf_train_pred), color='black', lw=2)
plt.legend(loc='upper left')
plt.show()

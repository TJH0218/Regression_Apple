import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# 讀取CSV資料
df = pd.read_csv('processed_apple_quality.csv')

# 選擇特徵和目標變數
X = df[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Acidity']]
y = df['Ripeness']

# 增加多項式特徵
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ridge 回歸模型
ridge_model = Ridge(alpha=1.0)  # 調整alpha參數
ridge_model.fit(X_train, y_train)
y_ridge_train_pred = ridge_model.predict(X_train)
y_ridge_test_pred = ridge_model.predict(X_test)
ridge_mse_train = mean_squared_error(y_train, y_ridge_train_pred)
ridge_r2_train = r2_score(y_train, y_ridge_train_pred)
ridge_mse_test = mean_squared_error(y_test, y_ridge_test_pred)
ridge_r2_test = r2_score(y_test, y_ridge_test_pred)
print("\nRidge Regression:")
print(f'Training MSE: {ridge_mse_train:.3f}')
print(f'Test MSE: {ridge_mse_test:.3f}')
print(f'Training R-squared: {ridge_r2_train:.3f}')
print(f'Test R-squared: {ridge_r2_test:.3f}')

# 繪製殘差圖
plt.figure(figsize=(8, 6))
plt.scatter(y_ridge_train_pred, y_ridge_train_pred - y_train, c='steelblue', edgecolor='white', marker='o', label='Training data')
plt.scatter(y_ridge_test_pred, y_ridge_test_pred - y_test, c='limegreen', edgecolor='white', marker='s', label='Test data')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Ridge Regression')
plt.hlines(y=0, xmin=min(y_ridge_train_pred), xmax=max(y_ridge_train_pred), color='black', lw=2)
plt.legend(loc='upper left')
plt.show()


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# 讀取CSV資料
df = pd.read_csv('processed_apple_quality.csv')

# 選擇特徵和目標變數
X = df[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Acidity']]
y = df['Ripeness']

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 添加多項式特徵
poly = PolynomialFeatures(degree=3)  # 這裡選擇3次多項式
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 線性回歸模型
lr_model = LinearRegression()
lr_model.fit(X_train_poly, y_train)
y_lr_train_pred = lr_model.predict(X_train_poly)
y_lr_test_pred = lr_model.predict(X_test_poly)
lr_mse_train = mean_squared_error(y_train, y_lr_train_pred)
lr_r2_train = r2_score(y_train, y_lr_train_pred)
lr_mse_test = mean_squared_error(y_test, y_lr_test_pred)
lr_r2_test = r2_score(y_test, y_lr_test_pred)
print("\nLinear Regression:")
print(f'Training MSE: {lr_mse_train:.3f}')
print(f'Test MSE: {lr_mse_test:.3f}')
print(f'Training R-squared: {lr_r2_train:.3f}')
print(f'Test R-squared: {lr_r2_test:.3f}')

# 繪製殘差圖
plt.figure(figsize=(8, 6))
plt.scatter(y_lr_train_pred, y_lr_train_pred - y_train, c='steelblue', edgecolor='white', marker='o', label='Training data')
plt.scatter(y_lr_test_pred, y_lr_test_pred - y_test, c='limegreen', edgecolor='white', marker='s', label='Test data')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Linear Regression')
plt.hlines(y=0, xmin=min(y_lr_train_pred), xmax=max(y_lr_train_pred), color='black', lw=2)
plt.legend(loc='upper left')
plt.show()


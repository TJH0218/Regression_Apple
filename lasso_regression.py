import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# 讀取CSV資料
df = pd.read_csv('processed_apple_quality.csv')

# 選擇特徵和目標變數
X = df[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Acidity']]
y = df['Ripeness']

# 增加多項式特徵
poly = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly.fit_transform(X)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 使用交叉驗證來選擇最佳的alpha值
lasso_model = LassoCV(cv=5, random_state=42, alphas=[0.1, 0.5, 1.0, 5.0, 10.0])
lasso_model.fit(X_train, y_train)
y_lasso_train_pred = lasso_model.predict(X_train)
y_lasso_test_pred = lasso_model.predict(X_test)
lasso_mse_train = mean_squared_error(y_train, y_lasso_train_pred)
lasso_r2_train = r2_score(y_train, y_lasso_train_pred)
lasso_mse_test = mean_squared_error(y_test, y_lasso_test_pred)
lasso_r2_test = r2_score(y_test, y_lasso_test_pred)
print("\nLasso Regression:")
print(f'Training MSE: {lasso_mse_train:.3f}')
print(f'Test MSE: {lasso_mse_test:.3f}')
print(f'Training R-squared: {lasso_r2_train:.3f}')
print(f'Test R-squared: {lasso_r2_test:.3f}')

# 繪製殘差圖
plt.figure(figsize=(8, 6))
plt.scatter(y_lasso_train_pred, y_lasso_train_pred - y_train, c='steelblue', edgecolor='white', marker='o', label='Training data')
plt.scatter(y_lasso_test_pred, y_lasso_test_pred - y_test, c='limegreen', edgecolor='white', marker='s', label='Test data')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Lasso Regression')
plt.hlines(y=0, xmin=min(y_lasso_train_pred), xmax=max(y_lasso_train_pred), color='black', lw=2)
plt.legend(loc='upper left')
plt.show()


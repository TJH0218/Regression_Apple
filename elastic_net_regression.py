import pandas as pd
from sklearn.linear_model import ElasticNet
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
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 調整Elastic Net 回歸模型的超參數
alpha = 0.1
l1_ratio = 0.5
en_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
en_model.fit(X_train, y_train)
y_en_train_pred = en_model.predict(X_train)
y_en_test_pred = en_model.predict(X_test)
en_mse_train = mean_squared_error(y_train, y_en_train_pred)
en_r2_train = r2_score(y_train, y_en_train_pred)
en_mse_test = mean_squared_error(y_test, y_en_test_pred)
en_r2_test = r2_score(y_test, y_en_test_pred)
print("\nElastic Net Regression:")
print(f'Training MSE: {en_mse_train:.3f}')
print(f'Test MSE: {en_mse_test:.3f}')
print(f'Training R-squared: {en_r2_train:.3f}')
print(f'Test R-squared: {en_r2_test:.3f}')

# 繪製殘差圖
plt.figure(figsize=(8, 6))
plt.scatter(y_en_train_pred, y_en_train_pred - y_train, c='steelblue', edgecolor='white', marker='o', label='Training data')
plt.scatter(y_en_test_pred, y_en_test_pred - y_test, c='limegreen', edgecolor='white', marker='s', label='Test data')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Elastic Net Regression')
plt.hlines(y=0, xmin=min(y_en_train_pred), xmax=max(y_en_train_pred), color='black', lw=2)
plt.legend(loc='upper left')
plt.show()


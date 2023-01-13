from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


df = pd.read_csv('advertising.csv')

sns.heatmap(df.corr(), cmap='Wistia', annot=True)

x = df.drop('Vendas', axis=1)
y = df['Vendas']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1
)

# criando a IA
lin_reg = LinearRegression()
rf_reg = RandomForestRegressor()

# treinando a IA
lin_reg.fit(x_train, y_train)
rf_reg.fit(x_train, y_train)

# testando a IA
test_pred_lin = lin_reg.predict(x_test)
test_pred_rf = rf_reg.predict(x_test)

r2_lin = metrics.r2_score(y_test, test_pred_lin)
rmse_lin = np.sqrt(metrics.mean_squared_error(y_test, test_pred_lin))

print(f'R² da regressão linear: {r2_lin}')
print(f'RSME da regressão linear: {rmse_lin}\n')

r2_rf = metrics.r2_score(y_test, test_pred_rf)
rmse_rf = np.sqrt(metrics.mean_squared_error(y_test, test_pred_rf))

print(f'R² do Random Forest: {r2_rf}')
print(f'RSME do Random Forest: {rmse_rf}')

df_resultado = pd.DataFrame()
df_resultado['y_teste'] = y_test
df_resultado['y_previsao_rf'] = test_pred_rf
df_resultado['y_previsão_lin'] = test_pred_lin
df_resultado = df_resultado.reset_index(drop=True)

fig = plt.figure(figsize=(15, 5))
sns.lineplot(data=df_resultado)
plt.show()
print(df_resultado)

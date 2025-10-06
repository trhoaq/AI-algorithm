import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os

feature1 = ['on road old','on road now','years','km','rating','condition','economy','hp','torque']
target = 'current price'

df = pd.read_csv("carCondition.csv", usecols=feature1+[target])
"""
sử dụng linear regression để dự đoán target theo feature
"""
x = df[feature1].values
y= df[target].values

X =  np.c_[np.ones(x.shape[0]),x]
w = np.linalg.inv(X.T @ X) @ X.T @ y

for i in range(len(feature1)):
    print(f"weight của {feature1[i]}: {w[i]}")

def pred(x_new, w):
    x_new = np.c_[np.ones(x_new.shape[0]),x_new]
    return x_new @ w

y_pred =  pred(x, w)

# plt.figure(figsize=(5, 5))
# plt.scatter(y, y_pred, alpha=0.3, c='blue', label='y_pred')
# plt.scatter(y, y_pred, c='red', alpha=0.3, label='y')
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) 
# plt.legend()
# plt.show()

"""
so sánh phần dư
"""
residuals = y - y_pred

plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.legend()
plt.show()

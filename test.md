# 演示文稿标题
## 副标题
    作者：Clay
    日期：2024年6月12日
---
## 目录

- 项目1
- 项目2
- 项目3

[这是一个链接](https://github.com)

---
## 第二页
使用 Markdown 添加更多内容。

---
## 代码高亮示例
```python [1|2|35|38]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 假设你已经有了一组实验数据
X = np.array([2,4,6,8,10]).reshape(-1, 1)  # 特征值
y = np.array([0.041,0.119,0.201,0.276,0.342])  # 目标值

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 打印回归方程的参数
print(f"回归方程: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

# 计算评估指标
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# 打印评估指标
print(f"均方误差 (MSE): {mse:.2f}")
print(f"决定系数 (R²): {r2:.2f}")

# 绘制原始数据点和拟合的直线
plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Fitted line')

# 显示图例
plt.legend()

# 显示图像
plt.show()
```
---
## 结论
在这里总结你的演示内容。

- 关键点1
- 关键点2
- 关键点3

---
## 感谢您的到来
Thanks for coming!
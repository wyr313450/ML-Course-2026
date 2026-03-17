# 实验二：概率建模、KNN分类与梯度下降优化（最终修复版）
# 适配环境：ml_env (PyTorch 2.10.0+cpu + numpy + matplotlib + scikit-learn)
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.neighbors import KNeighborsClassifier

# ===================== 全局配置：修复中文乱码 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False      # 正常显示负号

# ===================== 模块1：MLE与MAP参数估计 =====================
print("="*50)
print("模块1：MLE与MAP参数估计")
print("="*50)

# 1. 构造0-1样本数据并计算MLE、MAP
data = torch.tensor([1.,1.,0.,1.,0.])
p_mle = torch.mean(data)  # MLE核心公式
alpha = 2
beta_prior = 2  # 避免与scipy的beta重名
p_map = (torch.sum(data) + alpha - 1) / (len(data) + alpha + beta_prior - 2)  # MAP核心公式

# 打印结果
print(f"MLE = {p_mle.item():.4f}")
print(f"MAP = {p_map.item():.4f}\n")

# 2. 可视化：似然、先验、后验分布 + MLE/MAP标注
data_np = np.array([1,1,0,1,0])
N = len(data_np)
sum_x = np.sum(data_np)
p = np.linspace(0,1,100)  # 生成0-1之间100个点

# 计算似然、先验、后验
likelihood = p**sum_x * (1-p)**(N - sum_x)
prior = beta.pdf(p, alpha, beta_prior)  # Beta先验分布
posterior = p**(sum_x + alpha -1) * (1-p)**(N - sum_x + beta_prior -1)  # 后验分布

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(p, likelihood, label="Likelihood（似然）")
plt.plot(p, prior, label="Prior（Beta先验）")
plt.plot(p, posterior, label="Posterior（后验）")
plt.axvline(p_mle.item(), color='r', linestyle='--', label="MLE")
plt.axvline(p_map.item(), color='g', linestyle='--', label="MAP")
plt.legend(loc="best")
plt.title("MLE vs MAP (伯努利分布参数估计)")
plt.xlabel("参数p")
plt.ylabel("概率/密度")
plt.grid(alpha=0.3)
plt.show()

# ===================== 模块2：KNN分类 =====================
print("="*50)
print("模块2：KNN分类")
print("="*50)

# 1. 构造二维样本数据
X = np.array([[1,2],[2,3],[3,3],[6,5],[7,7],[8,6]])  # 特征
y = np.array([0,0,0,1,1,1])  # 标签（两类）

# 2. 生成网格点，用于绘制分类边界
xx, yy = np.meshgrid(np.linspace(0,10,200), np.linspace(0,10,200))
grid_points = np.c_[xx.ravel(), yy.ravel()]  # 展平为二维数组

# 3. 遍历不同K值，训练模型并绘图
for k in [1,3,5]:
    # 初始化并训练KNN模型
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X, y)
    # 预测网格点标签，重构为网格形状
    Z = knn_model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    # 绘图
    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")  # 分类边界
    plt.scatter(X[:,0], X[:,1], c=y, s=80, cmap="coolwarm", edgecolors="black")  # 样本点
    plt.title(f"KNN分类 (K = {k})")
    plt.xlabel("特征1")
    plt.ylabel("特征2")
    plt.grid(alpha=0.3)
plt.show()

# ===================== 模块3：梯度下降法 =====================
print("="*50)
print("模块3：梯度下降优化")
print("="*50)

# 目标函数：y = x² + 2x + 1（最小值在x=-1，y=0）
def target_fun(x):
    return x**2 + 2*x + 1

# 3.1 基础梯度下降：迭代20次，打印最终结果
x = torch.tensor([5.0], requires_grad=True)  # 初始值x=5，开启梯度追踪
lr = 0.1  # 学习率
for i in range(20):
    y = target_fun(x)
    y.backward()  # 反向传播计算梯度
    with torch.no_grad():  # 关闭梯度追踪，更新参数
        x -= lr * x.grad
    x.grad.zero_()  # 清空梯度，避免累积
print(f"最终参数x = {x.item():.4f}，最终损失y = {y.item():.4f}\n")

# 3.2 可视化1：损失函数变化曲线
loss_list = []
x = torch.tensor([5.0], requires_grad=True)
lr = 0.1
for i in range(20):
    y = target_fun(x)
    loss_list.append(y.item())
    y.backward()
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()

plt.figure(figsize=(8, 4))
plt.plot(loss_list, marker='o', markersize=4, color='blue')
plt.title("损失曲线 (梯度下降, 学习率=0.1)")
plt.xlabel("迭代次数")
plt.ylabel("损失值y")
plt.grid(alpha=0.3)
plt.show()

# 3.3 可视化2：梯度下降优化路径
x_vals = np.linspace(-5, 5, 100)
y_vals = target_fun(x_vals)  # 目标函数曲线
x_path = []  # 记录每次迭代的x值
x = torch.tensor([5.0], requires_grad=True)
lr = 0.3  # 增大学习率，让路径更明显
for i in range(10):
    y = target_fun(x)
    x_path.append(x.item())
    y.backward()
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()

# 绘制优化路径
plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_vals, color='blue', label="目标函数")
plt.scatter(x_path, [target_fun(xx) for xx in x_path], color='red', s=50, label="优化路径")
plt.axvline(x=-1, color='green', linestyle='--', label="最优值x=-1")
plt.title("梯度下降优化路径 (学习率=0.3)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("="*50)
print("所有模块运行完成！")
print("="*50)
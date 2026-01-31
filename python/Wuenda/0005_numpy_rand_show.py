import numpy as np
import matplotlib.pyplot as plt

# 展示一个圆环
def show_ring():
    # 随机生成10000个点，每个点的坐标在[-1, 1]之间
    data = 2 * np.random.rand(10000, 2) - 1
    # 提取第一列
    x = data[:,0]
    # 提取第二列
    y = data[:,1]
    # 筛选出在单位圆内的点
    idx = x**2 + y**2 <= 1
    # 筛选出在单位圆内但不在半径为0.25的圆内的点
    hole = x**2 + y**2 < 0.25
    # 抠出不在hole内的点，最终出来就是一个圆环
    idx = np.logical_and(idx, ~hole)
    plt.plot(x[idx], y[idx], "go", markersize=1)
    plt.show()

# 展示一个直方图
# 此示例展示了10000个随机数的分布情况，在0到1范围内基本是均匀分布
def barchart():
    p = np.random.rand(10000)
    np.set_printoptions(edgeitems=5000, suppress=True)
    plt.hist(p, bins=20, color='g', edgecolor='k')
    plt.show()

# 此示例展示的是100次随机数的平均值的分布情况，在0.5左右概率最高
def show_hist():
    N = 10000
    times = 100
    z = np.zeros(N)
    for i in range(times):
        z += np.random.rand(N)
    z /= times
    plt.hist(z, bins=20, color='m', edgecolor='k')
    plt.show()

# show_ring()
# barchart()
show_hist()
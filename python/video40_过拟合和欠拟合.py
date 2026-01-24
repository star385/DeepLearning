import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import common as cm

max_degree = 20
n_train, n_test = 100, 100

# 核心概念1：为什么这是多项式拟合？
# 数据生成公式：y = 5 + 1.2x - 3.4x^2 + 5.6x^3 + 噪声
# 这里 poly_features 就是 x 的各个次方 [x^0, x^1, x^2, ... x^19]
# 我们用线性模型 nn.Linear 去学习这些特征的权重，等价于学习多项式的系数
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
# 生成多项式特征：x -> [x^0, x^1, ..., x^19]
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!，用于归一化避免数值过大

labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

true_w, features, poly_features, labels = [
    torch.tensor(x, dtype=torch.float32, )
    for x in [true_w, features, poly_features, labels]
]
print(features[:2])
print(poly_features[:2, :])
print(labels[:2])

def evaluate_loss(net, data_iter, loss):
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel="epoch", ylabel="loss", yscale="log",
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=["train", "test"])
    
    for epoch in range(num_epochs):
        cm.train_net_ch3_using_framework(net, train_iter, loss, trainer, epoch)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
        print('weight:', net[0].weight.data.numpy())

# 核心概念2：不同拟合状态的对比
# 真实数据规律是3次曲线（需要前4个特征：x^0到x^3）

# 1. 正常拟合 (Normal)
# 使用前4个特征，模型复杂度 = 数据复杂度
# 就像让大学生考微积分，能力匹配，训练和测试误差都低
# train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
# d2l.plt.show()

# 2. 欠拟合 (Underfitting)
# 只使用前2个特征 (x^0, x^1)，即试图用直线去拟合曲线
# 就像让小学生考微积分，能力不足，学不会，训练和测试误差都高
train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
d2l.plt.show()

# 3. 过拟合 (Overfitting)
# 使用全部20个特征，模型复杂度 >> 数据复杂度
# 就像记忆力超群的书呆子，死记硬背了所有噪点，训练误差极低但测试误差高
# train(poly_features[:n_train, :], poly_features[n_train:, :],
#     labels[:n_train], labels[n_train:], num_epochs=1500)
# d2l.plt.show()

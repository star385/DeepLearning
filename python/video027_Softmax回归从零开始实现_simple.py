# video027_Softmax回归从零开始实现.py的简化版本，去除了调试代码，以及输出信息，这些信息占用了大量的篇幅，有点本末倒置，为学习增加了难度
import torch
from torchvision import transforms
import torchvision
from d2l import torch as d2l
import os

# 设置plt的参数，这个输出还是必要的，不然完全看不到运行效果也没有意义
d2l.plt.rcParams['font.sans-serif'] = ['SimHei']
d2l.plt.rcParams['figure.figsize'] = (10, 6)
d2l.plt.rcParams['font.size'] = 12

# 训练数据的存放目录，跟python平级的data目录
data_root = os.path.join(os.path.dirname(__file__), "../data")

# 训练数据的加载器，num_workers=0表示不使用多线程加载数据
# 老师的代码是返回4，但是我运行的时候会报错，调成0就没问题了，也许是电脑资源不足的原因
def get_dataloader_workers():
    return 0

def load_data_fashion_mnist(batch_size):
    trans = [transforms.ToTensor()]
    trans = transforms.Compose(trans)
    # 训练数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, transform=trans, download=True
    )
    # 测试数据集
    mnist_test = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, transform=trans, download=True
    )
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=get_dataloader_workers()))

batch_size = 256
# 加载数据集
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape((-1, num_inputs)), W) + b)

# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

def train_net_ch3(net, train_iter, loss, epoch, updater):
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        l.sum().backward()
        updater(X.shape[0])
        # 打印每个epoch的损失
        print(f"epoch {epoch}, loss {l.mean().item()}")

def train_ch3(net, train_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_net_ch3(net, train_iter, loss, epoch, updater)

lr = 0.1
def updater(batch_size):
    # 老师的代码是调用d2l.sgd，为了熟悉梯度下降，这里自己实现一下
    global W, b
    with torch.no_grad():
        W -= lr * W.grad / batch_size
        b -= lr * b.grad / batch_size
        W.grad.zero_()
        b.grad.zero_()

num_epochs = 20
train_ch3(net, train_iter, cross_entropy, num_epochs, updater)

# 如果为了纯粹的简化，下面的代码其实也没必要，但是我们总要看一下效果吧
def predict_ch3(net, test_iter, n=6):
    total, correct = 0, 0
    for X, y in test_iter:
        y_hat = net(X)
        y_pred = y_hat.argmax(axis=1)
        trues = d2l.get_fashion_mnist_labels(y)
        preds = d2l.get_fashion_mnist_labels(y_pred)
        total += len(y)
        correct += (y_pred == y).sum().item()
        print(f"预测结果: {preds[:n]}, 真实标签: {trues[:n]}")
    print(f"准确率: {correct * 1.0 / total}")

predict_ch3(net, test_iter)
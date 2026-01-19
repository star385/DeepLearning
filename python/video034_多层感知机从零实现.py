import torch
from torch import nn
import common as cm

batch_size = 256
                            
# 加载训练数据和测试数据集
train_iter, test_iter = cm.load_data_fashion_mnist(batch_size)

num_inputs, num_hiddens, num_outputs = 784, 256, 10

W1 = nn.Parameter(
    torch.randn(num_inputs, num_hiddens, requires_grad=True)
)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(
    torch.randn(num_hiddens, num_outputs, requires_grad=True)
)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

def relu(X):
    return torch.max(X, torch.zeros_like(X))

def net(X):
    X = X.reshape((-1, num_inputs))
    # @是一个什么乘法？
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)

loss = nn.CrossEntropyLoss()
num_epochs = 20
lr = 0.1
updater = torch.optim.SGD(params, lr=lr)

cm.train_ch3_from_scrach(net, train_iter, loss, updater, num_epochs)
cm.predict_fashion_mnist(net, test_iter)

# 实在搞不懂，为啥使用多层感知机，最后的准确率反倒还低？

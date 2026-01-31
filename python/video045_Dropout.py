# 这个翻译为丢弃法或者暂退法
import torch
from torch import nn
from d2l import torch as d2l
import common as cm

# X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
# print(X)
# print(dropout_layer(X, 0.))
# print(dropout_layer(X, 0.5))
# print(dropout_layer(X, 1.))

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = cm.load_data_fashion_mnist(batch_size)

def impl_from_scratch():
    def dropout_layer(X, dropout):
        assert 0 <= dropout <= 1
        # 如果的dropout为1，那么所有元素都被丢弃，返回全0张量
        if dropout == 1:
            return torch.zeros_like(X)
        
        # 如果dropout为0，那么所有元素都被保留，返回原张量
        if dropout == 0:
            return X
        
        # 随机选择一些元素设置为0
        mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()
        return mask * X / (1.0 - dropout)

    class Net(nn.Module):

        def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
            is_training=True):
            super(Net, self).__init__()
            self.num_inputs = num_inputs
            self.num_outputs = num_outputs
            self.is_training = is_training
            self.lin1 = nn.Linear(num_inputs, num_hiddens1)
            self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
            self.lin3 = nn.Linear(num_hiddens2, num_outputs)
            self.relu = nn.ReLU()

        def forward(self, X):
            H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
            if self.is_training:
                H1 = dropout_layer(H1, dropout1)
            H2 = self.relu(self.lin2(H1))
            if self.is_training:
                H2 = dropout_layer(H2, dropout2)
            return self.lin3(H2)

    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    cm.train_ch3_using_framework(net, train_iter, loss, trainer, num_epochs)
    cm.predict_fashion_mnist(net, test_iter)

def impl_using_framework():
    # 简洁实现
    net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_hiddens1), nn.ReLU(),
        nn.Dropout(dropout1), nn.Linear(num_hiddens1, num_hiddens2), nn.ReLU(),
        nn.Dropout(dropout2), nn.Linear(num_hiddens2, num_outputs))

    def init_weights(m):
        if (type(m) == nn.Linear):
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    cm.train_ch3_using_framework(net, train_iter, loss, trainer, num_epochs)
    cm.predict_fashion_mnist(net, test_iter)

# impl_from_scratch()
impl_using_framework()
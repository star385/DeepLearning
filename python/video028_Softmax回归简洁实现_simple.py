import torch
from torch import nn
from d2l import torch as d2l
import os
import torchvision
from torchvision import transforms

batch_size = 256
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

def get_dataloader_workers():
    return 0

def load_data_fashion_mnist(batch_size):
    trans = [transforms.ToTensor()]
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, transform=trans, download=True
    )
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                        num_workers=get_dataloader_workers()),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                        num_workers=get_dataloader_workers()))
                            
# 加载训练数据和测试数据集
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_outputs = 10
net = nn.Sequential(nn.Flatten(), nn.Linear(784, num_outputs))

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weight)

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

def train_net_ch3(net, train_iter, loss, updater, epoch):
    net.train()
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.backward()
        updater.step()
        print(f"epoch {epoch}, loss {l.item()}")

def train_ch3(net, train_iter, loss, updater, epoch):
    for epoch in range(epoch):
        train_net_ch3(net, train_iter, loss, updater, epoch)

num_epochs = 20
train_ch3(net, train_iter, loss, trainer, num_epochs)

def predict_ch3(net, test_iter):
    total, correct = 0, 0
    for X, y in test_iter:
        y_hat = net(X)
        y_pred = y_hat.argmax(dim=1)
        # preds = d2l.get_fashion_mnist_labels(y_pred)
        # trues = d2l.get_fashion_mnist_labels(y)
        correct += (y_pred == y).sum().item()
        total += y.numel()
    print(f"total:{total}, correct:{correct}, accuracy: {correct / total}")

predict_ch3(net, test_iter)

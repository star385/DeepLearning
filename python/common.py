import torchvision
from torchvision import transforms
import torch
import os

# 数据集根目录
data_root = os.path.join(os.path.dirname(__file__), '../data')

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


# 从零开始训练
def __train_net_ch3_common(net, train_iter, loss, updater, epoch):
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            print(f"epoch {epoch}, loss {l.mean().item()}")
        else:
            l.sum().backward()
            updater(X.shape[0])
            print(f"epoch {epoch}, loss {l.mean().item()}")
    pass


# 从零开始训练
def train_net_ch3_from_scratch(net, train_iter, loss, updater, epoch):
    __train_net_ch3_common(net, train_iter, loss, updater, epoch)


def train_net_ch3_using_framework(net, train_iter, loss, updater, epoch):
    net.train()
    __train_net_ch3_common(net, train_iter, loss, updater, epoch)

def train_ch3_from_scrach(net, train_iter, loss, updater, num_epochs):
    for epoch in range(num_epochs):
        train_net_ch3_from_scratch(net, train_iter, loss, updater, epoch)

def train_ch3_using_framework(net, train_iter, loss, updater, num_epochs):
    for epoch in range(num_epochs):
        train_net_ch3_using_framework(net, train_iter, loss, updater, epoch)

def predict_fashion_mnist(net, test_iter):
    total, correct = 0, 0
    for X, y in test_iter:
        y_hat = net(X)
        y_pred = y_hat.argmax(dim=1)
        correct += (y_pred == y).sum().item()
        total += y.numel()
    print(f"total:{total}, predict correct:{correct} accuracy: {correct / total}")

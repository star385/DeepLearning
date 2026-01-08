import os
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
trans = transforms.ToTensor()

# 获取当前脚本所在目录的上一级目录下的 data 文件夹路径
# 这样无论在哪个目录下运行脚本，数据都会下载到项目根目录下的 data 文件夹
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

mnist_train = torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root=data_root, train=False, transform=trans, download=True)
# print(len(mnist_train))
# print(len(mnist_test))
print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels):
    """ 返回Fashion-MNIST数据集的文本标签 """
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot',
    ]
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """ 绘制图像列表 """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# 使用plt显示前18张图片
# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
# d2l.plt.show()

batch_size = 256
def get_dataloader_workers():
    """ 使用4个进程来读取数据 """
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue

print(f'读取训练集数据时间: {timer.stop():.2f}秒')

def load_data_fashion_mnist(batch_size, resize=None):
    """ 下载Fashion-MNIST数据集，然后将其加载到内存中 """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, transform=trans, download=True
    )
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True, 
            num_workers=get_dataloader_workers()),
        data.DataLoader(mnist_test, batch_size, shuffle=False,
            num_workers=get_dataloader_workers())
    )

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, y.shape, y.dtype)
    break
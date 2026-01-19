import torch
from torch import nn
import common as cm

num_inputs, num_hiddens, num_outputs = 784, 256, 10

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

traine_iter, test_iter = cm.load_data_fashion_mnist(batch_size)
cm.train_ch3_using_framework(net, traine_iter, loss, trainer, num_epochs)

cm.predict_fashion_mnist(net, test_iter)
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import loadDataset as load
import matplotlib.pyplot as plt

class model(nn.Module):
    def __init__(self, n_dim, n_class):
        self.W = np.random.randn(n_dim, n_class)  # initial random weight: W.shape = (10, 3072)
        self.b = np.zeros(n_class)

    def softmax(self, y):  # input y is output of the model y = xW + b
        denominator = np.sum(np.exp(y), axis=1)[:, np.newaxis]
        return np.exp(y) / denominator

    def loss(self, y_true, y_pred):
        return np.mean(np.sum(-y_true * np.log(y_pred), axis=1))

    def onehot(self, y, n_class):
        vectors = np.zeros((len(y), n_class))
        for i, label in enumerate(y):
            vectors[i, label] = 1
        return vectors

    def forward(self, x):
        y = np.dot(x, self.W) + self.b
        return self.softmax(y)

def run():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    xtr, ytr, xte, yte = load.load_CIFAR10()
    xtr = xtr.reshape(xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
    xte = xte.reshape(xte.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072

    myModel = model(xtr.shape[1], 10)
    mini_batch = xtr[:4, :]
    output = myModel.forward(mini_batch)
    print('shape of output =', output.shape)  # each row assigns 10 numbers
    #print(output)
    output = myModel.softmax(output)

    trained = myModel.loss(myModel.onehot(ytr[:4], 10), output)
    print("real label: ", ytr[:4])
    print("train loss: %.5f", trained)
    print("predict label:", np.argmax(output, axis=1))

    y_pred = myModel.forward(xte)
    y_true = myModel.onehot(yte, 10)
    test_loss = myModel.loss(y_true, y_pred)
    print('Test loss = %.5f' % test_loss)


if __name__ == '__main__':
    run()
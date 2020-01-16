from __future__ import print_function
import gzip
import numpy as np
import matplotlib.pyplot as plt
import random

key_file = {
    'train_x':'train-images-idx3-ubyte.gz',
    'train_y':'train-labels-idx1-ubyte.gz',
    'test_x':'t10k-images-idx3-ubyte.gz',
    'test_y':'t10k-labels-idx1-ubyte.gz'
}

#dataset directory
dataset_dir = "/home/mech-user/B3_3A/robotintelligence/data/mnist"

dataset = {}
#unzip data and exchange to array
for k in key_file.keys():
    if 'x' in k:
        file_path = dataset_dir + '/' + key_file[k]
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 784)
        dataset[k] = data

    elif 'y' in k:
        file_path = dataset_dir + '/' + key_file[k]
        with gzip.open(file_path, 'rb') as f:
            label = np.frombuffer(f.read(), np.uint8, offset=8)
        dataset[k]  = label

#normalization
train_x = dataset['train_x'] / 255
test_x = dataset['test_x'] / 255
#one-hot
train_y = np.eye(10)[dataset['train_y']].astype(np.int32)
test_y = np.eye(10)[dataset['test_y']].astype(np.int32)
#dataset number
train_num = train_x.shape[0]
test_num = test_x.shape[0]
pixel_value = train_x.shape[1]

#define activation function
#ReLU function
class ReLU:
    def __init__(self):
        self.x = None
        self.param = False

    def __call__(self, x, train_config=True):
        #forwoard propagation
        self.x = x
        return x * (x > 0)

    def backward(self, dout):
        #back propagation
        return dout * (self.x > 0)

#sigmoid function
class Sigmoid:
    def __init__(self):
        self.y = None
        self.param = False

    def __call__(self, x, train_config=True):
        #forwoard propagation
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, dout):
        #back propagation
        return dout * self.y * (1 - self.y)

#Softmax function
class Softmax:
    def __init__(self):
        self.x = None
        self.y = None
        self.param = False

    def __call__(self, x, train_config=True):
        self.x = x
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))
        y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.y = y
        return y

#define Linear Layer
class Linear:
    def __init__(self, input_unit, output_unit, init_weight):
        if init_weight == 'std':
            #random initial number of weight
            self.W = 0.01 * np.random.randn(input_unit, output_unit)
        #He init value if use ReLu
        elif init_weight == 'He':
            scale =  np.sqrt(2.0 / input_unit)
            self.W = scale * np.random.randn(input_unit, output_unit)

        #initialize bias
        self.b = np.zeros(output_unit)
        self.delta = None
        self.x = None
        self.dW = None
        self.db = None
        self.param = True
        self.name = 'Linear'

    def __call__(self, x, train_config=True):
        #forward propagation
        self.x = x
        y =  np.dot(x, self.W) + self.b
        return y

    def backward(self, delta):
        #back propagation
        dout =  np.dot(delta, self.W.T)
        #gradient calculation
        self.dW =  np.dot(self.x.T, delta)
        self.db = np.dot(np.ones(len(self.x)), delta)
        return dout

#define optimizer
#SGD
class SGD():
    def __init__(self, lr):
        self.lr = lr
        self.network = None

    def setup(self, network):
        self.network = network

    def update(self):
        for layer in self.network.layers:
            if layer.param:
                layer.W -=  self.lr * layer.dW
                layer.b -=   self.lr * layer.db
#MomentumSGD
class MomentumSGD():
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        self.network = None

    def setup(self, network):
        self.network = network
        self.v = {'W': [], 'b': []}
        for layer in self.network.layers:
            if layer.param:
                self.v['W'].append(np.zeros_like(layer.W))
                self.v['b'].append(np.zeros_like(layer.b))

    def update(self):
        layer_idx = 0
        for layer in self.network.layers:
            if layer.param:
                self.v['W'][layer_idx] =  self.momentum * self.v['W'][layer_idx] - self.lr * layer.dW
                self.v['b'][layer_idx] = self.momentum * self.v['b'][layer_idx] - self.lr * layer.db

                layer.W += self.v['W'][layer_idx]
                layer.b += self.v['b'][layer_idx]

                layer_idx += 1

#for dropout
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.param = False

    def __call__(self, x, train_config=True):
        if train_config:
            self.mask =  np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return  dout * self.mask

#define Multilayer Perceptron
class MultilayerPerceptron():
    def __init__(self, layers, init_weight='std'):
        self.layers = layers
        self.t = None

    def forward(self, x, t, train_config=True):
        self.t = t
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y, train_config)
        self.loss =  np.sum(-t*np.log(self.y + 1e-7)) / len(x)
        return self.loss

    def backward(self):
        dout = (self.y - self.t) / len(self.layers[-1].x)
        for layer in self.layers[-2::-1]:
            dout =  layer.backward(dout)

optimizer_SGD = SGD(lr=0.5)
model_3_Sigmoid = MultilayerPerceptron([Linear(784, 1000, init_weight='std'),
                         Sigmoid(),
                         Dropout(),
                         Linear(1000, 1000, init_weight='std'),
                         Sigmoid(),
                         Dropout(),
                         Linear(1000, 10, init_weight='std'),
                         Softmax()])

"""
model_3_Sigmoid = MultilayerPerceptron([Linear(784, 1000, init_weight='std'),
                         Sigmoid(),
                         #Dropout(),
                         Linear(1000, 1000, init_weight='std'),
                         Sigmoid(),
                         #Dropout(),
                         Linear(1000, 10, init_weight='std'),
                         Softmax()])
"""

optimizer_SGD.setup(model_3_Sigmoid)

model_3_ReLU = MultilayerPerceptron([Linear(784, 1000, init_weight='He'),
                         ReLU(),
                         Dropout(),
                         Linear(1000, 1000, init_weight='He'),
                         ReLU(),
                         Dropout(),
                         Linear(1000, 10, init_weight='He'),
                         Softmax()])

optimizer_momentum = MomentumSGD(lr=0.1, momentum=0.9)
optimizer_momentum.setup(model_3_Sigmoid)

model_4 = MultilayerPerceptron([Linear(784, 1000, init_weight='He'),
                         ReLU(),
                         Dropout(),
                         Linear(1000, 1000, init_weight='He'),
                         ReLU(),
                         Dropout(),
                         Linear(1000, 100, init_weight='He'),
                         ReLU(),
                         Dropout(),
                         Linear(100, 10, init_weight='He'),
                         Softmax()])

#optimizer_momentum.setup(model_4)
#optimizer_SGD.setup(model_4)

def train(model, optimizer, noise_ratio):
    epoch_num = 20
    batchsize = 100
    epoch_list = []
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    #add noise
    for i in range(train_num):
        for j in range(pixel_value):
            if random.random() <= noise_ratio:
                train_x[i][j] = random.random()

    for epoch in range(1, epoch_num+1):
        print('epoch {} | '.format(epoch), end="")

        epoch_list.append(epoch)

        #train
        sum_loss = 0
        pred_y = []
        perm = np.random.permutation(train_num)

        for i in range(0, train_num, batchsize):
            x = train_x[perm[i: i+batchsize]]
            t = train_y[perm[i: i+batchsize]]

            loss = model.forward(x, t)
            model.backward()
            optimizer.update()

            sum_loss += loss * len(x)
            pred_y.extend(np.argmax(model.y, axis=1).tolist())

        loss = sum_loss / train_num
        accuracy = np.sum(np.eye(10)[pred_y] * train_y[perm]) / train_num
        print('Train loss {}, accuracy {} | '.format(float(loss), accuracy), end="")
        train_loss.append(float(loss))
        train_acc.append(accuracy)

        sum_loss = 0

        pred_y = []
        for i in range(0, test_num, batchsize):
            x = test_x[i: i+batchsize]
            t = test_y[i: i+batchsize]

            sum_loss += model.forward(x, t, train_config=False) * len(x)
            pred_y.extend(np.argmax(model.y, axis=1).tolist())
        loss = sum_loss / test_num

        accuracy = np.sum(np.eye(10)[pred_y] * test_y) / test_num
        print('Test loss {}, accuracy {}'.format(float(loss), accuracy))
        test_loss.append(float(loss))
        test_acc.append(accuracy)

    return train_loss, train_acc, test_loss, test_acc, epoch_list

train_loss, train_acc, test_loss, test_acc, epoch_list = train(model_3_Sigmoid, optimizer_SGD, 0.1) #set model optimizer noise_ratio

plt.title("noise 10%")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.plot(epoch_list, test_acc)
plt.show()

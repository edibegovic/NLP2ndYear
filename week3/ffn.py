
from sklearn.metrics import confusion_matrix
from pandas.core.common import flatten
import matplotlib.pyplot as plt
from collections import Counter
from keras import backend as K
from keras  import activations
from numpy.random import randn
from collections import defaultdict
import numpy as np
import pandas as pd
import re
import pickle

def keras_softmax(scores):
    ## softmax calculation
    var = K.variable(value=scores)
    act_tf = activations.softmax(var) # returns Tensor
    softmax_scores = K.eval(act_tf) # return numpy array
    return softmax_scores

data = [('da','Nielsen'),
        ('da','Jensen'),
        ('da','Hansen'),
        ('da','Pedersen'),
        ('da','Andersen'),
        ('se','Andersson'),
        ('se','Johansson'),
        ('se','Karlsson'),
        ('se','Nilsson'),
        ('se','Eriksson'),
        ('no','Hansen'), 
        ('no','Johansen'),
        ('no','Olsen'),
        ('no','Larsen'),
        ('no','Andersen'),
       ]

labels = [label2idx[l] for l, _ in data]
names = [x for _, x in data]

char2idx = {char:idx for idx, char in enumerate(set("".join(names)))}
label2idx = {"da": 0,"se": 1, "no": 2}

# Vocab size
len(char2idx)

# N-hot encoding of data
x_train = np.zeros((len(names), len(char2idx)))
for i, w in enumerate(names):
    for char in w:
        x_train[i, char2idx[char]] += 1

# Parameters
input_dim = 20
h1_dim = 14
h2_dim = 19
output_dim = 3

w1 = randn(h1_dim, input_dim)
w2 = randn(h2_dim, h1_dim)
w3 = randn(output_dim, h2_dim)
b2 = 0
b3 = 0

# FNN
a1 = np.matmul(w1, x_train[0])
z1 = np.tanh(a1)
a2 = np.matmul(w2, z1) + b2
z2 = np.tanh(a2)
a3 = np.matmul(w3, z2) + b3

y_hat_manual = keras_softmax(a3.reshape(1,3))

# y_hat sums to 1
np.sum(y_hat_manual)

# Use model with real weigts
with open("data/weights.pickle","rb") as f:
    weights = pickle.load(f)

data_eval = x_train

# Keras implementation
from keras.models import load_model
model = load_model('data/model.h5')
predictions = model.predict(data_eval)
print(predictions)

# My implementation
w1 = weights["W1"]
w2 = weights["W2"]
w3 = weights["W3"]
b1 = weights["b1"]
b2 = weights["b2"]
b3 = weights["b3"]

# FNN
a1 = np.matmul(data_eval, w1) + b1
z1 = np.tanh(a1)
a2 = np.matmul(z1, w2) + b2
z2 = np.tanh(a2)
a3 = np.matmul(z2, w3) + b3
y_hat_manual = keras_softmax(a3)
y_hat_manual

# Predicted labels
np.argmax(y_hat_manual, axis=1)

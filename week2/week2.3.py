
import numpy as np
import re
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
from pandas.core.common import flatten

regex = r'([\w\'\-]+(\.\w+)*)'
token = re.compile(regex)
first = lambda x: [a[0] for a in x]

x = []
y = []

with open("data/train.star.txt", "r") as f: 
    texts = f.read().split("\n")
    for t in texts:
        temp = t.split("\t")
        if len(temp) > 1:
            y.append(temp[0])
            x.append(first(token.findall(temp[1])))

x = np.array(x)
y = np.array(y).reshape(-1, 1)

vocab = {key: value for (value, key) in enumerate(set(flatten(x)))}
train_x = np.zeros((len(x), len(vocab)))

for idx, line in enumerate(x):
    for w in line:
        if w in vocab:
            train_x[idx, vocab[w]] += 1

# Perceptron
clf = Perceptron(tol=1e-3)
clf.fit(train_x, y)

# Testing
x_t = []
y_t = []

with open("data/dev.other.txt", "r") as f: 
    texts = f.read().split("\n")
    for t in texts:
        temp = t.split("\t")
        if len(temp) > 1:
            y_t.append(temp[0])
            x_t.append(first(token.findall(temp[1])))

x_t = np.array(x_t)
y_t = np.array(y_t).reshape(-1, 1)

dev_x = np.zeros((len(x_t), len(vocab)))

for idx, line in enumerate(x_t):
    for w in line:
        if w in vocab:
            dev_x[idx, vocab[w]] += 1

y_pred = clf.predict(dev_x)

accuracy_score(y_t, y_pred)
confusion_matrix(y_t, y_pred)

flipped_vocab = {value:key for key, value in vocab.items()}
for i in range(10):
    print(flipped_vocab[np.argsort(clf.coef_[0])[::-1][i]])



from collections import defaultdict 
from keras import utils
import keras
import numpy as np
import re
import tensorflow
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

PAD = " <pad> "
window_size = 2
first = lambda x: [a[0] for a in x]

def add_padding(docs, padding = 2):
    return [padding*PAD+doc+padding*PAD for doc in docs]

def list_of_words(docs):
    flattend_docs = ' '.join(docs)
    regex = r'([\w\'\-<>]+(\.\w+)*)'
    token = re.compile(regex)
    words = first(token.findall(flattend_docs))
    return words

# courpus = ["this is an example", "another example", "I love deep learning"]
with open('sample.txt','r') as f:
    courpus = f.read().split('\n')

docs_with_padding = add_padding(courpus)
word_list = list_of_words(docs_with_padding)

# Fix this mess
unique_words = [PAD.strip()]+(list(set(word_list)-{PAD.strip()}))
word2idx = {word:idx for idx, word in enumerate(unique_words)}

train_X = []
train_y = []

for i, w in enumerate(word_list):
    if w == PAD.strip():
        continue
    window = word_list[i-2: i]+word_list[i+1: i+3]
    train_X.append([word2idx[a] for a in window])
    train_y.append(word2idx[w])

# One-hot encoding
train_y = keras.utils.to_categorical(train_y)

# ------------------------ Task 2 ------------------------

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda

cbow = Sequential()
cbow.add(Embedding(input_dim=len(word2idx), output_dim=64, input_length=4))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(vocab_size, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# view model summary
print(cbow.summary())

from scipy.spatial.distance import cosine

def get_sim(weights, vec):
    dists = [(idx, cosine(w, vec)) for idx, w in enumerate(weights)]
    sorted_dists = sorted(dists, key=lambda x: x[1])
    print(first(sorted_dists)[:10])


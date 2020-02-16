
from collections import Counter
import matplotlib.pyplot as plt
import re
import pandas as pd
from pandas.core.common import flatten
import numpy as np

df_train = pd.read_csv('data/metal-lyrics-train.csv')

# Deliverable 1.1
def bag_of_words(text):
    regex = r'([\w\'\-]+(\.\w+)*)'
    token = re.compile(regex)
    first = lambda x: [a[0] for a in x]
    return Counter(first(token.findall(text)))

def read_data(filename, label='Era', text="Lyrics", preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values, [preprocessor(string) for string in df[text].values]

y_train, x_train = read_data('data/metal-lyrics-train.csv')
y_dev, x_dev = read_data('data/metal-lyrics-dev.csv')

# Deliverable 1.2
plt.hist(y_train)
plt.show()

# Deliverable 1.3
def aggregate_counts(bags_of_words):
    raw_text = " ".join(list(flatten(bags_of_words)))
    return bag_of_words(raw_text)

# Deliverable 1.4
def compute_oov(bow1, bow2):
    return set(bow2.keys())-set(bow1.keys())

def oov_rate(bow1, bow2):
    return len(compute_oov(bow1, bow2)) / len(bow1.keys())

# Deliverable 1.5
oov_rate(aggregate_counts(x_train), aggregate_counts(x_dev))

# Deliverable 1.5.b
not_in_train = len(compute_oov(aggregate_counts(x_train), aggregate_counts(x_dev)))
not_in_train/len(set(aggregate_counts(x_dev).keys()))
# 22%-ish

# Power law
plt.loglog([val for word, val in aggregate_counts(x_train).most_common()])
plt.loglog([val for word, val in aggregate_counts(x_dev).most_common()])
plt.legend(['training set','dev set']);
plt.show()

######### TASK 2 NAIVE BAYES #########

# Deliverable 1.4
def aggregate_counts_for_label(bags_of_words, y, label):
    indecis = np.argwhere(y == label)
    return aggregate_counts(np.take(bags_of_words, indecis))

# Tiny test
y_tiny_train, x_tiny_train = read_data('data/tinysentiment_train.csv', label = "Label", text = "Text")
y_tiny_dev, x_tiny_dev = read_data('data/tinysentiment_test.csv', label = "Label", text = "Text")


#############################################
#############  TO BE IGNORED  ###############
#############################################
from abc import ABC

class LinearClassifier(ABC):

    def __init__(self):
        self.trained = False
        self.lab2idx = {} # mapping of each class label to an index
        self.feat2idx = {} # mapping of features to indices

    def train(self, X, y):
        raise NotImplementedError

    def get_scores(self, x, w):
        return np.dot(x, w)

    def get_label(self, x, w):
        scores = np.dot(x, w)
        return np.argmax(scores, axis=1).transpose()

    def test(self, x, w):
        if not self.is_trained:
            raise Error("Please train the model first")
        idx2lab = {i: lab for lab, i in self.lab2idx.items()} # reverse mapping
        x_matrix = np.zeros((len(x),len(self.feat2idx)+1)) # add prior
        for i, inst in enumerate(x):
            # add prior
            for j, p_c in enumerate(w[0]):
                x_matrix[i][0] = 1
            # likelihood
            for f in inst:
                if f in self.feat2idx: #otherwise ignore
                    fIdx = self.feat2idx[f]
                    x_matrix[i][fIdx] = inst[f]
               
        predicted_label_indices = self.get_label(x_matrix, w)
        return [idx2lab[i] for i in predicted_label_indices]

    def evaluate(self, gold, predicted):
        correct = 0
        total = 0
        for g,p in zip(gold,predicted):
            if g == p:
                correct += 1
            total += 1
        return correct/total
#############################################
#############################################

# Deliverable 2.1
class NaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.is_trained = False

    def train(self, X, y):
        print("Training a multinomial NB model")
        params = self.train_nb(X, y)
        self.is_trained = True
        return params
    
    ### deliverable 2.1
    def train_nb(self, X_train, y_train):
        # estimate the model parameters
        
        # this function should return the following matrix
        # 
        #   parameters = np.zeros((vocab_size+1, num_classes))
        #
        #   where
        #    - the first row [0] contains the prior (log probs) per class  parameters[0, i] = log p of c
        #    - and the remaining rows contain the per class likelihood parameters[1:, i]
        #
           
        num_classes = len(np.unique(y_train))
        features_train = aggregate_counts(X_train)
        vocab_size = len(features_train)
        print("{} classes, {} vocab size".format(num_classes, vocab_size))

        # instantiate mappers
        self.feat2idx = {f: i+1 for i,f in enumerate(features_train)}  # keep 0 reserved for prior 
        self.lab2idx = {l: i for i,l in enumerate(np.unique(y_train))}


        # your code here
        
        
        #return parameters
        raise NotImplementedError

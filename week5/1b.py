
from sklearn.metrics import accuracy_score
import numpy as np

# A -------------------------------------

with open('17120.pos.txt','r') as f:
    peer_POS = f.read().split("\n")

with open('new_17120.pos.txt','r') as f:
    og_POS = f.read().split("\n")

def is_new_tweet(text):
    return not (text == '' or text.startswith('# text'))

filter_tweets = lambda a: list(filter(is_new_tweet, a))
split_tags = lambda a: np.array(list(map(lambda b: b.split(), a))).T

_, edi_tags = split_tags(filter_tweets(og_POS))
_, peer_tags = split_tags(filter_tweets(peer_POS))

accuracy_score(edi_tags, peer_tags)
# Accuracy is 0.718
# Mostly due to me not having marked @usernames
# as PRON but instead X.

# B -------------------------------------

def confusion_matrix(y_true, y_pred):
    match = lambda t, p: len([_ for a, b in zip(y_true, y_pred) if a==t and b==p])
    true_dist = lambda t: np.array([match(t, p) for p in set(y_true)])
    return np.array([true_dist(t) for t in set(y_true)])

def kappa_score(l1, l2):
    N = len(l1)
    conf_mat = confusion_matrix(l1, l2)
    A_0 = np.trace(conf_mat)/N
    A_e = np.sum((np.sum(conf_mat, axis=0)/N)*(np.sum(conf_mat, axis=1)/N))
    return (A_0-A_e)/(1-A_e)

kappa_score(edi_tags, peer_tags)
# Kappe score is 0.689

# C -------------------------------------
# X INTJ
# PRON / X
# PART, SCONJ



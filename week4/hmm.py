
import codecs
import numpy as np
import myutils
from collections import defaultdict
from operator import itemgetter

# load data
train_data = myutils.read_conll_file("data/da_ddt-ud-train.conllu")
dev_data = myutils.read_conll_file("data/da_ddt-ud-dev.conllu")

#############################################
########## HMM CLASS GOES HERE ##############
#############################################
# Copy paste it from the ipynb...
#############################################
#############################################

def predict_most_likely(self, sentence):
    token_probs = lambda t: [(tag, self.emissions[tag][t]) for tag in self.tags]
    most_likely_tag = lambda t: max(token_probs(t), key=itemgetter(1))[0]
    return [most_likely_tag(token) for token in sentence]

# To make the function link to the class in the previous cell
HMM.predict_most_likely = predict_most_likely


# Test naiive
hmm = HMM()
hmm.fit(train_data)
most_likely_predictions = hmm.predict(dev_data, method='most_likely')
gold = [x[1] for x in dev_data]
sent_level, word_level = myutils.evaluate(gold, most_likely_predictions)
print('most likely scores:')
print('sent level:  {:.4f}'.format(sent_level))
print('word level:  {:.4f}'.format(word_level))
print()




import sys
import nose
import numpy as np
import pandas as pd
from importlib import reload

! nosetests tests/test_environment.py

df_train = pd.read_csv('data/corpus.csv')
df_train.head()

! cat data/corpus.csv | wc -l

assert(len(df_train)==7)

# ----------------------------------
# 1.1

from snlp import preproc

reload(preproc);
x_train = preproc.read_data('data/corpus.csv',preprocessor=preproc.space_tokenizer)

! nosetests tests/test_preproc.py:test_space_tok


# ----------------------------------
# 1.2

reload(preproc);
! nosetests tests/test_preproc.py:test_create_vocab

print(preproc.create_vocab(x_train))

# ----------------------------------
# 2.1 

from snlp import lm
reload(lm);
x_train = preproc.read_data('data/corpus.csv',preprocessor=preproc.space_tokenizer)

# ----------------------------------
# 2.2

# instantiate a uniform LM 
vocab = preproc.create_vocab(x_train)
uniformLM = lm.UniformLM(vocab)

# get size of vocab and probabilities of words
for word in vocab:
    print(uniformLM.probability(word), word)

# ----------------------------------
# 2.3

lm.sample(uniformLM, ["the"], 10)

# ----------------------------------
# 2.4

x_dev = preproc.read_data('data/corpus_dev.csv',preprocessor=preproc.space_tokenizer)
lm.perplexity(uniformLM, x_dev)


# ----------------------------------
# 3.1

reload(lm);
! nosetests tests/test_lms.py:test_unigram

# ----------------------------------
# 3.2

Reference prev. exercise

# ----------------------------------
# 3.3

unigramLM = lm.UnigramLM(vocab, x_train)
for word in vocab:
    print(unigramLM.probability(word), word)

lm.perplexity(uniformLM, x_dev)
lm.perplexity(unigramLM, x_dev)

# ----------------------------------
# 4.1

!cat data/corpus.csv

reload(lm);
! nosetests tests/test_lms.py:test_bigram

reload(lm);
unigramLM = lm.NgramLM(vocab, x_train, 2)
unigramLM.counts(("is", "a"))
unigramLM.norm(("is",))
unigramLM.probability(("is"))

# ----------------------------------
# 5.1

vocab = preproc.create_vocab(x_train)
unigramLM = lm.UnigramLM(vocab, x_train)
print(unigramLM.vocab)

print(unigramLM.probability("the"))
print(unigramLM.probability("blue"))

# Having a fixed vocabulary beforehand
# and introducce <UNK> token for rest
# or
# simply intruduce <UNK> for words that
# appear less than some threshold.

# for test-sets we add smoothening.

# ----------------------------------
# 6.1

# (2+1) / (3+11) = 0.214




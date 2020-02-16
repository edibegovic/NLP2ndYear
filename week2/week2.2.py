
# ----------------------------------
# 2.4.1

# P(doc|pos): 0.09 * 0.07 * 0.29 * 0.04 * 0.08 = 5.85 * 10e-6
# P(doc|neg): 0.16 * 0.06 * 0.06 * 0.15 * 0.11 = 9.50 * 10e-6

# We ignore prior prob as they are the same
# Would classify as negative

# ----------------------------------
# 2.4.2

# P(category | doc) = P(category) * ((doc_1_count + 1)/(total_unq_words + total_words_in_category))

# P(comedy | doc) = (2/5) * ((1+1)/(9+7)) * ((2+1)/(9+7)) * ((0+1)/(9+7)) * ((1+1)/(9+7))
# P(action | doc) = (3/5) * ((2+1)/(11+7)) * ((0+1)/(11+7)) * ((4+1)/(11+7)) * ((1+1)/(11+7))

# doc belongs to action

# ----------------------------------
# 2.4.3

# Multinomial
# P(pos|doc) = (2/5) * ((3+1)/(9+3))**2 * ((1+1)/(9+3)) * ((5+1)/(9+3))
# P(neg|doc) = (3/5) * ((2+1)/(14+3))**2 * ((2+1)/(14+3)) * ((10+1)/(14+3))
# Classified as POSITIVE

# Binary
# P(pos|doc) = (2/5) * ((1+1)/(4+3)) * ((1+1)/(4+3)) * ((2+1)/(4+3))
# P(neg|doc) = (3/5) * ((2+1)/(6+3)) * ((3+1)/(6+3)) * ((1+1)/(6+3))
# Classified as NEGATIVE


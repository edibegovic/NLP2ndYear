

#----------- 1 RegEx -----------

# a)
# [csm]at[\s,."?!)]
# \b[csm]at\b

# b)
# (\b[\w]+\b)\s\1\b

# c)
# [\d]+[\d,\.]+(\s(kr|DKK))?


#----------- 2. Tokenization -----------

new_regex = r'[\w\-]+(\'\w+)*|[.?,;:!]'

# 1) Improving tokenization
# Q: The tokenizer clearly misses out parts of the text. Which?
# A: paranteser, 'word (apostrophe as quotes), varoius punctiuation (?!,), combined-words

# Q: Should one separate 'm, 'll, n't, possessives, and other forms of contractions from the word?
# A: No - one looses contex

# Q: Should elipsis be considered as three '.'s or one '...'?
# A: As a single group. That is more than two dots should belong to the "same group".

# Q: There's a bunch of these small rules - will you implement all of them to create a 'perfect' tokenizer?
# A: Until some point of diminishing returns. But not all.


# 2) Twitter

twitter_regex = r'http:\/\/[\w.\/#]+|[@#\w]+'

# Q: What does 'correctly' mean, when it comes to Twitter tokenization?
# A: Words, hashtags, @users, emojis, links

# Q: What defines correct tokenization of each tweet element?
# A:
# Hastag: #-sign followed by letters
# Mention: @-sign followed by letters
# Links: groups of charects starting with "http://" encapsulated by whitespace

# Q: How will your tokenizer tokenize elipsis (...)?
# A: \.{2, }

# Q: Will it correctly tokenize emojis?
# A: Real ones? 😀 Yes. ¯\_(ツ)_/¯ <- No.

# Q: What about composite emojis?
# A: No. 


#----------- 3. Segmentation -----------

import re

# 1)
def sentence_segment(match_regex, tokens):
    current = []
    sentences = [current]
    for tok in tokens:
        current.append(tok)
        if match_regex.match(tok):
            current = []
            sentences.append(current)
    if not sentences[-1]:
        sentences.pop(-1)
    return sentences

# 2)
text = """
Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is the longest official one-word placename in U.K. Isn't that weird? I mean, someone took the effort to really make this name as complicated as possible, huh?! Of course, U.S.A. also has its own record in the longest name, albeit a bit shorter... This record belongs to the place called Chargoggagoggmanchauggagoggchaubunagungamaugg. There's so many wonderful little details one can find out while browsing http://www.wikipedia.org during their Ph.D. or an M.Sc.
"""

regex = r'(http:\/\/[\.\/\w]*|[\w\'\-]+(\.\w+)*|[.,?!])'
token = re.compile(regex)
first = lambda x: [a[0] for a in x]

tokens = first(token.findall(text))
sentences = sentence_segment(re.compile('\.'), tokens)
for sentence in sentences:
    print(sentence)

# QUESTIONS
# Q: What elements of a sentence did you have to take care of here?
# A: Links, words, punctuation (,.?!), abbrevations(?) ("U.S.A"), coupled-words(?)

# Q: Is it useful or possible to enumerate all such possible examples?
# A: Nope


# Q: How would you deal with all URLs effectively?
# Capture them as whole tokens


# Q: Are there any specific punctuation not covered in the example you might think of?
# A: *OMG*




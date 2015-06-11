#!/bin/python
# coding=utf-8

import nltk.corpus
from nltk.corpus import wordnet
import nltk.stem.snowball
import string
import sys

nltk.data.path.append('./nltk_data/')

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')

tokenizer = nltk.tokenize.word_tokenize
stemmer = nltk.stem.snowball.SnowballStemmer('english')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def token(t):
    """http://en.wikipedia.org/wiki/Tokenization_(lexical_analysis)"""
    return [token.lower() for token in tokenizer(t)]

def stopword(t):
    """http://en.wikipedia.org/wiki/Stop_words"""
    return [x.strip(string.punctuation) for x in token(t) if x not in stopwords]

def stem(t):
    """http://en.wikipedia.org/wiki/Stemming"""
    return map(stemmer.stem, stopword(t))

def pos_tag(t):
    """http://en.wikipedia.org/wiki/Part_of_speech"""
    return nltk.pos_tag(stopword(t))

def lemma(t):
    """http://en.wikipedia.org/wiki/Lemmatisation"""
    def get_wordnet_pos(pos_tag):
        if pos_tag[1].startswith('J'):
            return (pos_tag[0], wordnet.ADJ)
        elif pos_tag[1].startswith('V'):
            return (pos_tag[0], wordnet.VERB)
        elif pos_tag[1].startswith('N'):
            return (pos_tag[0], wordnet.NOUN)
        elif pos_tag[1].startswith('R'):
            return (pos_tag[0], wordnet.ADV)
        else:
            return (pos_tag[0], wordnet.NOUN)

    wordnet_pos = map(get_wordnet_pos, pos_tag(t))
    return [lemmatizer.lemmatize(x, pos) for x, pos in wordnet_pos if pos == wordnet.NOUN]

def ner(t):
    """http://en.wikipedia.org/wiki/Named-entity_recognition"""
    return nltk.ne_chunk(pos_tag(t))

def main(text):
    print '\n'.join([
        'Original: {}'.format(text),
        'Token: {}'.format(token(text)),
        'Stopwords: {}'.format(stopword(text)),
        'Stem: {}'.format(stem(text)),
        'Part of Speech: {}'.format(pos_tag(text)),
        'Lemma: {}'.format(lemma(text)),
        'NER: {}'.format(ner(text)),
    ])

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print 'Usage: python nlp.py "When will Ida have her next period?"'
        sys.exit(1)

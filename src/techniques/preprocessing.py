""" techniques.preprocessing module """

import string
import re
from nltk import word_tokenize, pos_tag
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def get_tokenized_list(doc_text):
    """ Converts a string into a tokenized list """
    return word_tokenize(doc_text)

def word_stemmer(token_list):
    """ Performs stemming on a tokenized list """
    if not isinstance(token_list, list):
        raise TypeError
    ps = PorterStemmer()
    stemmed = []
    for words in token_list:
        stemmed.append(ps.stem(words))
    return stemmed

def pos_tagger(token_list):
    """ Perform POS tagging on a tokenized list """
    return pos_tag(token_list)

def remove_stopwords(token_list):
    """ Perform stop word removal on a tokenized list """
    if not isinstance(token_list, list):
        raise TypeError
    cleaned_token_list = []
    for word in token_list:
        if word.lower() not in ENGLISH_STOP_WORDS:
            cleaned_token_list.append(word)
    return cleaned_token_list

def remove_punctuation(doc_text: str):
    """ Perform punctuation removal on a string """
    return doc_text.translate(str.maketrans('', '', string.punctuation))

def retrieve_corpus(us_dataset):
    """ Build an user story corpus """
    corpus = []
    for entry in us_dataset:
        corpus.append(entry["preprocessed_text"])
    return corpus

def get_us_action(us: str) -> str:
    """ Return only the action of the user story """
    try:
        return re.search(r'i want(.*?)so that', us, flags=re.IGNORECASE | re.S).group(1).strip()
    except AttributeError:
        pass
    try:
        return re.search(r'i want(.*?)$', us, flags=re.IGNORECASE | re.S).group(1).strip()
    except AttributeError:
        # TODO: log message
        return us

def remove_us_skeleton(us: str) -> str:
    """ Remove the user story template words """
    skeleton = ["as a ", "as an ", "i want ", "so that ", "*as* ", "*i want* ", "*As*\n"]
    skeleton = "|".join(map(re.escape, skeleton))
    compiled = re.compile("(%s)" % skeleton, flags=re.IGNORECASE | re.S)
    us = compiled.sub("", us)
    return us

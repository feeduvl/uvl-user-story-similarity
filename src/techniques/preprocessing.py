import string
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
    corpus = []
    for entry in us_dataset:
        corpus.append(entry["text"])
    return corpus

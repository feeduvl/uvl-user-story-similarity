from nltk import word_tokenize, pos_tag
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def get_tokenized_list(doc_text):
    return word_tokenize(doc_text)

def word_stemmer(token_list):
    ps = PorterStemmer()
    stemmed = []
    for words in token_list:
        stemmed.append(ps.stem(words))
    return stemmed

def pos_tagger(token_list):
    return pos_tag(token_list)

def remove_stopwords(doc_text):
    cleaned_text = []
    for words in doc_text:
        if words not in ENGLISH_STOP_WORDS:
            cleaned_text.append(words)
    return cleaned_text

def retrieve_corpus(us_dataset):
    corpus = []
    for entry in us_dataset:
        corpus.append(entry["text"])
    return corpus

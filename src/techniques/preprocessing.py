import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def get_tokenized_list(doc_text):
    return nltk.word_tokenize(doc_text)

def word_stemmer(token_list):
    ps = nltk.stem.PorterStemmer()
    stemmed = []
    for words in token_list:
        stemmed.append(ps.stem(words))
    return stemmed

def remove_stopwords(doc_text):
    cleaned_text = []
    for words in doc_text:
        if words not in ENGLISH_STOP_WORDS:
            cleaned_text.append(words)
    return cleaned_text


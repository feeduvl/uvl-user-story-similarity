import nltk
from nltk.corpus import stopwords

def get_stopwords():
    return set(stopwords.words('english'))

def get_tokenized_list(doc_text):
    return nltk.word_tokenize(doc_text)

def word_stemmer(token_list):
    ps = nltk.stem.PorterStemmer()
    stemmed = []
    for words in token_list:
        stemmed.append(ps.stem(words))
    return stemmed

def remove_stopwords(doc_text):
    stop_words = get_stopwords()
    cleaned_text = []
    for words in doc_text:
        if words not in stop_words:
            cleaned_text.append(words)
    return cleaned_text
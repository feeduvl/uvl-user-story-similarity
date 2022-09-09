from src.techniques.userStorySimilarity import UserStorySimilarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.mock.mockData import corpus
from src.techniques.preprocessing import get_tokenized_list, remove_stopwords, word_stemmer

class UserStorySimilarityVsm(UserStorySimilarity):
    def measure_similarity(self, us_dataset):
        corpus = self.retrieve_corpus(us_dataset)
        preprocessed_docs = self.perform_preprocessing(corpus)
        vectorizer = TfidfVectorizer()
        vectorizer.fit(preprocessed_docs)
        doc_vector = vectorizer.transform(preprocessed_docs)
        cosine_similarities = cosine_similarity(doc_vector)

        # store results
        # all_pairs = {}  # TODO: could be a nice way to store all results, if needed
        result = []
        for i in range(len(cosine_similarities)):
            for j in range(i):
                # keys = frozenset({us_dataset[i]["id"], us_dataset[j]["id"]})
                # all_pairs[keys] = cosine_similarities[i][j]
                self.processResult(cosine_similarities[i][j], result, us_dataset[i], us_dataset[j])

        return result
        
    def processResult(self, score, result, us1, us2):
        if score > 0.5:  # TODO: Remove hardcoded threshold
            result_entry = {
                "id_1": us1["id"],
                "id_2": us2["id"],
                "us_text_1": us1["text"],
                "us_text_2": us2["text"],
                "score": score,
                "ac_1": us1["acceptance_criteria"],
                "ac_2": us2["acceptance_criteria"],
                "raw_text_1": us1["raw_text"],
                "raw_text_2": us2["raw_text"]
            }
            result.append(result_entry)

    def retrieve_corpus(self, us_dataset):
        corpus = []
        for entry in us_dataset:
            corpus.append(entry["text"])
        return corpus
            

    def perform_preprocessing(self, corpus):
        preprocessed_corpus = []
        for doc in corpus:
            tokens = get_tokenized_list(doc)
            doc_text = remove_stopwords(tokens)
            doc_text  = word_stemmer(doc_text)
            doc_text = ' '.join(doc_text)
            preprocessed_corpus.append(doc_text)
        return preprocessed_corpus

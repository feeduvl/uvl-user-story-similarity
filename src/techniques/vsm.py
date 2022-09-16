import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.techniques.preprocessing import (get_tokenized_list, remove_stopwords,
                                          word_stemmer)
from src.techniques.userStorySimilarity import UserStorySimilarity


class UserStorySimilarityVsm(UserStorySimilarity):

    def measure_all_pairs_similarity(self, us_dataset):
        corpus = self.retrieve_corpus(us_dataset)
        preprocessed_docs = self.perform_preprocessing(corpus)
        vectorizer = TfidfVectorizer()
        vectorizer.fit(preprocessed_docs)
        doc_vector = vectorizer.transform(preprocessed_docs)
        cosine_similarities = cosine_similarity(doc_vector)

        # store results
        result = self.processResultAllPairs(cosine_similarities, us_dataset)
        return result

    # TODO: handle case when besides the focused user story there is no other
    def measure_pairwise_similarity(self, us_dataset: list, focused_id):
        focused_index = next((i for i, item in enumerate(us_dataset) if item["id"] == focused_id), None)
        if focused_index is None:
            # if the ID does not exist or if the user story could not be extracted
            # TODO: return error in the metrics of api response
            return []

        corpus = self.retrieve_corpus(us_dataset)
        preprocessed_docs = self.perform_preprocessing(corpus)
        preprocessed_query = self.perform_preprocessing([corpus[focused_index]])
        # TODO: check if the same
        # preprocessed_query = self.perform_preprocessing_on_query(corpus[focused_index])
        # preprocessed_query = [preprocessed_query]
        vectorizer = TfidfVectorizer()
        vectorizer.fit(preprocessed_docs)
        doc_vector = vectorizer.transform(preprocessed_docs)
        query_vector = vectorizer.transform(preprocessed_query)
        cosine_similarities = cosine_similarity(doc_vector, query_vector).flatten()

        # remove focused user story
        focused_user_story = us_dataset[focused_index]
        del us_dataset[focused_index]
        cosine_similarities = np.delete(cosine_similarities, focused_index)

        # store results
        result = self.processResultFocused(cosine_similarities, us_dataset, focused_user_story)

        return result
        
    def processResultFocused(self, cosine_similarities, us_dataset, focused_user_story):
        sim_result = []
        for i in range(len(cosine_similarities)):
            if cosine_similarities[i] > 0.5:  # TODO: Remove hardcoded threshold
                result_entry = {
                    "id": us_dataset[i]["id"],
                    "us_text": us_dataset[i]["text"],
                    "score": cosine_similarities[i],
                    "ac": us_dataset[i]["acceptance_criteria"],
                    "raw_text": us_dataset[i]["raw_text"]
                }
                sim_result.append(result_entry)

        result = {
            "focused": focused_user_story,
            "similar_user_stories": sim_result
        }
        return result
 
    def processResultAllPairs(self, cosine_similarities, us_dataset):
        result = []
        # all_pairs = {}  # TODO: could be a nice way to store all results, if needed
        for i in range(len(cosine_similarities)):
            for j in range(i):
                # keys = frozenset({us_dataset[i]["id"], us_dataset[j]["id"]})
                # all_pairs[keys] = cosine_similarities[i][j]
                if cosine_similarities[i][j] > 0.5:
                    result_entry = {
                        "id_1": us_dataset[i]["id"],
                        "id_2": us_dataset[j]["id"],
                        "us_text_1": us_dataset[i]["text"],
                        "us_text_2": us_dataset[j]["text"],
                        "score": cosine_similarities[i][j],
                        "ac_1": us_dataset[i]["acceptance_criteria"],
                        "ac_2": us_dataset[j]["acceptance_criteria"],
                        "raw_text_1": us_dataset[i]["raw_text"],
                        "raw_text_2": us_dataset[j]["raw_text"]
                    }
                    result.append(result_entry)
        
        return result

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

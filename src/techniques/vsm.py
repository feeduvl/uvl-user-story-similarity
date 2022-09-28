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
        doc_vector = vectorizer.fit_transform(preprocessed_docs)
        cosine_similarities = cosine_similarity(doc_vector)

        # store results
        result = self.process_result_all_pairs(cosine_similarities, us_dataset)
        return result

    # TODO: handle case when besides the focused user story there is no other
    def measure_pairwise_similarity(self, us_dataset: list, focused_ids: list[str]):
        corpus = self.retrieve_corpus(us_dataset)
        preprocessed_docs = self.perform_preprocessing(corpus)
        vectorizer = TfidfVectorizer()
        doc_vector = vectorizer.fit_transform(preprocessed_docs) # TODO: Consider preprocessor, tokenizer, stop-words from this vectorizer
        cosine_similarities = []
        result = []
        finished_indices = []

        for focused_id in focused_ids:
            focused_index = next((i for i, item in enumerate(us_dataset) if item["id"] == focused_id), None)
            if focused_index is None:
                # if the ID does not exist or if the user story could not be extracted
                # TODO: return error in the metrics of api response
                continue

            preprocessed_query = [preprocessed_docs[focused_index]]
       
            query_vector = vectorizer.transform(preprocessed_query)
            cosine_similarities_focused = cosine_similarity(doc_vector, query_vector).flatten().tolist()
            cosine_similarities.append(cosine_similarities_focused)

            self.process_result_entry_focused(cosine_similarities_focused, us_dataset, focused_index, finished_indices, result)

            finished_indices.append(focused_index)

        return result

    def process_result_entry_focused(self, cosine_similarities_focuesd, us_dataset, focused_index, finished_indices, result):
        focused_user_story = us_dataset[focused_index]
        for i, (score, us_representation) in enumerate(zip(cosine_similarities_focuesd, us_dataset)):
            if i == focused_index or i in finished_indices:
                continue
            self.map_to_us_representation(focused_user_story, us_representation, score, result)

    def map_to_us_representation(self, first, second, score, result_entries):
        if score > 0.5:
            result_entry = {
                "id_1": first["id"],
                "id_2": second["id"],
                "us_text_1": first["text"],
                "us_text_2": second["text"],
                "score": score,
                "ac_1": first["acceptance_criteria"],
                "ac_2": second["acceptance_criteria"],
                "raw_text_1": first["raw_text"],
                "raw_text_2": second["raw_text"]
            }
            result_entries.append(result_entry)
 
    def process_result_all_pairs(self, cosine_similarities, us_dataset):
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

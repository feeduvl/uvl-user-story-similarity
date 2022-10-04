import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.techniques.preprocessing import (get_tokenized_list, remove_stopwords,
                                          word_stemmer, retrieve_corpus)
from src.techniques.userStorySimilarity import UserStorySimilarity
from src.feedUvlMapper import FeedUvlMapper


class UserStorySimilarityVsm(UserStorySimilarity):

    def __init__(self, feed_uvl_mapper: FeedUvlMapper, threshold: float) -> None:
        self.feed_uvl_mapper = feed_uvl_mapper
        self.threshold = threshold

    def measure_all_pairs_similarity(self, us_dataset):
        corpus = retrieve_corpus(us_dataset)
        preprocessed_docs = self.perform_preprocessing(corpus)
        vectorizer = TfidfVectorizer()
        doc_vector = vectorizer.fit_transform(preprocessed_docs)
        cosine_similarities = cosine_similarity(doc_vector).tolist()

        # store results
        result = self.process_result_all_pairs(cosine_similarities, us_dataset)
        return result

    # TODO: handle case when besides the focused user story there is no other
    def measure_pairwise_similarity(self, us_dataset: list, focused_ids: list[str]):
        corpus = retrieve_corpus(us_dataset)
        preprocessed_docs = self.perform_preprocessing(corpus)
        vectorizer = TfidfVectorizer()
        doc_vector = vectorizer.fit_transform(preprocessed_docs) # TODO: Consider preprocessor, tokenizer, stop-words from this vectorizer
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
            self.process_result_entry_focused(cosine_similarities_focused, us_dataset, focused_index, finished_indices, result)
            finished_indices.append(focused_index)

        return result

    def process_result_all_pairs(self, cosine_similarities, us_dataset):
        result = []

        for i, (score_row, us_representation_1) in enumerate(zip(cosine_similarities[:-1], us_dataset[:-1])):
            for score, us_representation_2 in zip(score_row[i+1:], us_dataset[i+1:]):
                self.feed_uvl_mapper.map_to_us_representation(us_representation_1, us_representation_2, score, result, self.threshold)
        
        return result

    def process_result_entry_focused(self, cosine_similarities_focuesd, us_dataset, focused_index, finished_indices, result):
        focused_user_story = us_dataset[focused_index]
        for i, (score, us_representation) in enumerate(zip(cosine_similarities_focuesd, us_dataset)):
            if i == focused_index or i in finished_indices:
                continue
            self.feed_uvl_mapper.map_to_us_representation(focused_user_story, us_representation, score, result, self.threshold) 

    def perform_preprocessing(self, corpus):
        preprocessed_corpus = []
        for doc in corpus:
            tokens = get_tokenized_list(doc)
            doc_text = remove_stopwords(tokens)
            doc_text  = word_stemmer(doc_text)
            doc_text = ' '.join(doc_text)
            preprocessed_corpus.append(doc_text)
        return preprocessed_corpus

""" techniques.vsm module """

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.feeduvl_mapper import FeedUvlMapper
from src.techniques.preprocessing import (get_tokenized_list, remove_punctuation, remove_stopwords,
                                          retrieve_corpus, word_stemmer)
from src.techniques.user_story_similarity import UserStorySimilarity


class UserStorySimilarityVsm(UserStorySimilarity):
    """
    User Story similarity analysis with the Vector Space Model and TF-IDF weighting
    using the cosine similarity as metric
    """

    def __init__(self, feed_uvl_mapper: FeedUvlMapper, threshold: float) -> None:
        self.feed_uvl_mapper = feed_uvl_mapper
        self.threshold = threshold

    def measure_all_pairs_similarity(self, us_dataset: list):
        """ Similarity analysis for all pairwise user story combinations """
        corpus = retrieve_corpus(us_dataset)
        preprocessed_docs = self._perform_preprocessing(corpus)
        vectorizer = TfidfVectorizer()
        if not preprocessed_docs:
            return []
        doc_vector = vectorizer.fit_transform(preprocessed_docs)
        cosine_similarities = cosine_similarity(doc_vector).tolist()

        # store results
        result = self._process_result_all_pairs(cosine_similarities, us_dataset)
        return result

    # TODO: handle case when besides the focused user story there is no other
    def measure_pairwise_similarity(self, us_dataset: list, focused_ids: list[str], unextracted_ids: list[str]):
        """
        Similarity analysis for all focused user stories\n
        The user stories given in focused focused_ids are compared to every other user story in the dataset
        """
        corpus = retrieve_corpus(us_dataset)
        preprocessed_docs = self._perform_preprocessing(corpus)
        vectorizer = TfidfVectorizer()
        doc_vector = vectorizer.fit_transform(preprocessed_docs) # TODO: Consider preprocessor, tokenizer, stop-words from this vectorizer

        result = []
        finished_indices = []
        unexistent_ids_count = 0
        for focused_id in focused_ids:
            focused_index = next((i for i, item in enumerate(us_dataset) if item["id"] == focused_id), None)
            if focused_index is None:
                # the ID does not exist or the user story could not be extracted
                if focused_id not in unextracted_ids:
                    unexistent_ids_count += 1
                continue

            preprocessed_query = [preprocessed_docs[focused_index]]
            query_vector = vectorizer.transform(preprocessed_query)
            cosine_similarities_focused = cosine_similarity(doc_vector, query_vector).flatten().tolist()
            self._process_result_entry_focused(cosine_similarities_focused, us_dataset, focused_index, finished_indices, result)
            finished_indices.append(focused_index)

        return result, unexistent_ids_count

    def _process_result_all_pairs(self, cosine_similarities, us_dataset):
        result = []

        for i, (score_row, us_representation_1) in enumerate(zip(cosine_similarities[:-1], us_dataset[:-1])):
            for score, us_representation_2 in zip(score_row[i+1:], us_dataset[i+1:]):
                self.feed_uvl_mapper.map_similarity_result(us_representation_1, us_representation_2, score, self.threshold, result)

        return result

    def _process_result_entry_focused(self, cosine_similarities_focuesd, us_dataset, focused_index, finished_indices, result):
        focused_user_story = us_dataset[focused_index]
        for i, (score, us_representation) in enumerate(zip(cosine_similarities_focuesd, us_dataset)):
            if i == focused_index or i in finished_indices:
                continue
            self.feed_uvl_mapper.map_similarity_result(focused_user_story, us_representation, score, self.threshold, result)

    def _perform_preprocessing(self, corpus):
        preprocessed_corpus = []
        for doc in corpus:
            doc_text = remove_punctuation(doc)
            tokens = get_tokenized_list(doc_text)
            doc_text = remove_stopwords(tokens)
            doc_text  = word_stemmer(doc_text)
            doc_text = ' '.join(doc_text)
            preprocessed_corpus.append(doc_text)
        return preprocessed_corpus

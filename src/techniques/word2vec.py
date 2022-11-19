""" techniques.word2vec module """

import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from src.techniques.user_story_similarity import UserStorySimilarity
from src.techniques.preprocessing import (get_tokenized_list, remove_punctuation,
    remove_stopwords, retrieve_corpus, remove_us_skeleton, get_us_action)
from src.feeduvl_mapper import FeedUvlMapper

class UserStorySimilarityWord2vec(UserStorySimilarity):
    """
    User Story similarity analysis using a pretrained Word2vec model
    together with cosine similarity and the scoring formular from mihalcea
    """
    model = None

    def __init__(self, feed_uvl_mapper: FeedUvlMapper, threshold: float, without_us_skeleton: bool, only_us_action: bool) -> None:
        self.feed_uvl_mapper = feed_uvl_mapper
        self.threshold = threshold
        self.without_us_skeleton = without_us_skeleton
        self.only_us_action = only_us_action
        self.idf_of_tokens = None
        self.unique_tokens = None

    @staticmethod
    def load_model(logger: logging.Logger):
        """ load a pretrained model which is used for similarity analysis with Word2vec """
        logger.info("Loading word2vec model ...")
        try:
            # solutions to memory error:
            #   increase RAM -> (8GB should be sufficient)
            #   consider limit param (e.g. limit=500000) -> 1/6 of oriinal size
            #   use another model with less features, less feature size
            UserStorySimilarityWord2vec.model: KeyedVectors = KeyedVectors.load_word2vec_format(
                'data/GoogleNews-vectors-negative300-SLIM.bin', binary=True
            )
            logger.info("Done loading word2vec model.")
        except MemoryError:
            logger.error("Error when loading word2vec model. Probably not enough RAM.")


    def measure_all_pairs_similarity(self, us_dataset):
        """ Similarity analysis for all pairwise user story combinations """
        corpus = retrieve_corpus(us_dataset)
        preprocessed_corpus, preprocessed_docs =self._perform_preprocessing(corpus)
        result = []
        if not preprocessed_docs or len(preprocessed_docs) == 1:
            return result

        vectorizer = TfidfVectorizer()
        self.unique_tokens = vectorizer.fit(preprocessed_docs).get_feature_names_out()
        self.idf_of_tokens = vectorizer.idf_

        for i, (us_representation_1, preprocessed_corpus_element_1) in enumerate(zip(us_dataset[:-1], preprocessed_corpus[:-1])):
            for us_representation_2, preprocessed_corpus_element_2 in zip(us_dataset[i+1:], preprocessed_corpus[i+1:]):
                score = self._user_story_similarity(preprocessed_corpus_element_1, preprocessed_corpus_element_2)
                self.feed_uvl_mapper.map_similarity_result(us_representation_1, us_representation_2, score, self.threshold, result)

        return result

    def measure_pairwise_similarity(self, us_dataset: list, focused_ids: list[str], unextracted_ids: list[str]):
        """
        Similarity analysis for all focused user stories\n
        The user stories given in focused focused_ids are compared to every other user story in the dataset
        """
        corpus = retrieve_corpus(us_dataset)
        preprocessed_corpus, preprocessed_docs =self._perform_preprocessing(corpus)
        result = []
        if not preprocessed_docs or len(preprocessed_docs) == 1:
            return result
        vectorizer = TfidfVectorizer()
        self.unique_tokens = vectorizer.fit(preprocessed_docs).get_feature_names_out()
        self.idf_of_tokens = vectorizer.idf_

        finished_indices = []
        unexistent_ids_count = 0
        for focused_id in focused_ids:
            focused_index = next((i for i, item in enumerate(us_dataset) if item["id"] == focused_id), None)
            if focused_index is None:
                # the ID does not exist or the user story could not be extracted
                if focused_id not in unextracted_ids:
                    unexistent_ids_count += 1
                continue

            focused_us_representation = us_dataset[focused_index]
            focused_corpus_element = preprocessed_corpus[focused_index]
            for i, (us_representation_2, preprocessed_corpus_element_2) in enumerate(zip(us_dataset, preprocessed_corpus)):
                if i == focused_index or i in finished_indices:
                    continue
                score = self._user_story_similarity(focused_corpus_element, preprocessed_corpus_element_2)
                self.feed_uvl_mapper.map_similarity_result(focused_us_representation, us_representation_2, score, self.threshold, result)

            finished_indices.append(focused_index)

        return result, unexistent_ids_count

    def _user_story_similarity(self, user_story_1, user_story_2):
        numerator_1, denominator_1 = self._calculate_term(user_story_1, user_story_2)
        numerator_2, denominator_2 = self._calculate_term(user_story_2, user_story_1)

        score = 0.5*(numerator_1/denominator_1 + numerator_2/denominator_2)
        return score

    def _calculate_term(self, user_story_1, user_story_2):
        numerator = 0.0
        denominator = 0.0
        for word_1 in user_story_1:
            try:
                UserStorySimilarityWord2vec.model[word_1]
            except KeyError:
                continue
            token_index = next((i for i, token in enumerate(self.unique_tokens) if token == word_1.lower()), None)
            if token_index is None:
                logging.warning(f'Token index for word {word_1} is not found. This should not occur.')
                continue
            idf_of_token = self.idf_of_tokens[token_index]

            best_score = 0
            scores = []
            for word_2 in user_story_2:
                try:
                    score = UserStorySimilarityWord2vec.model.similarity(word_1, word_2)
                    scores.append(score)
                except KeyError:
                    continue
            best_score = max(scores)
            if best_score is not None:
                best_score *= idf_of_token
                numerator += best_score
                denominator += idf_of_token
        return numerator, denominator

    def _perform_preprocessing(self, corpus):
        preprocessed_corpus = []
        preprocessed_docs = []
        for doc in corpus:
            # corpus preprocessing
            doc_text = remove_punctuation(doc)
            if self.only_us_action:
                doc_text = get_us_action(doc_text)
            elif self.without_us_skeleton:
                doc_text = remove_us_skeleton(doc_text)
            tokens = get_tokenized_list(doc_text)
            tokens = remove_stopwords(tokens)
            preprocessed_doc_text = ' '.join(tokens)
            preprocessed_corpus.append(tokens)
            preprocessed_docs.append(preprocessed_doc_text)

        return preprocessed_corpus, preprocessed_docs

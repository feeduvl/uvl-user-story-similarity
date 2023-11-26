""" techniques.use module """


import tensorflow as tf

import tensorflow_hub as hub

import logging
import configuration

from src.feeduvl_mapper import FeedUvlMapper
from src.techniques.preprocessing import (get_tokenized_list, remove_punctuation, remove_stopwords,
                                          retrieve_corpus, word_stemmer, remove_us_skeleton, get_us_action)
from src.techniques.user_story_similarity import UserStorySimilarity


class UserStorySimilarityUse(UserStorySimilarity):
    """
    User Story similarity analysis with DL based LM 'BERT'.
    """

    def __init__(self, feed_uvl_mapper: FeedUvlMapper, threshold: float, without_us_skeleton: bool, only_us_action: bool, no_preprocessing: bool) -> None:
        self.feed_uvl_mapper = feed_uvl_mapper
        self.threshold = threshold
        self.without_us_skeleton = without_us_skeleton
        self.only_us_action = only_us_action
        self.no_preprocessing = no_preprocessing

        self.modelname = "Universal Sentence Encoder"
        self.model = None
    
    def __del__(self):
        logging.debug(msg=f"Cleaning up references for DL model {self.modelname} to free ressources.")
        del self.model
        self.model = None

    def measure_all_pairs_similarity(self, us_dataset: list):
        """ Similarity analysis for all pairwise user story combinations """
        if len(us_dataset) <= 1:
            return []
        corpus = retrieve_corpus(us_dataset)
        preprocessed_docs = self.__perform_preprocessing(corpus)
        
        data = preprocessed_docs
        self.__setup_model()
        embeddings = self.__create_embeddings(data)
        cosine_similarities = self.__calculate_cosine_similarities(embeddings)

        # store results
        result = self.__process_result_all_pairs(cosine_similarities, us_dataset)
        return result

    def measure_pairwise_similarity(self, us_dataset: list, focused_ids: list[str], unextracted_ids: list[str]):
        """
        Similarity analysis for all focused user stories\n
        The user stories given in focused focused_ids are compared to every other user story in the dataset
        """
        result = []
        finished_indices = []
        unexistent_ids_count = 0
        if len(us_dataset) <= 1:
            return result, unexistent_ids_count
        corpus = retrieve_corpus(us_dataset)
        preprocessed_docs = self.__perform_preprocessing(corpus)
        
        data = preprocessed_docs
        self.__setup_model()
        embeddings = self.__create_embeddings(data)

        for focused_id in focused_ids:
            focused_index = next((i for i, item in enumerate(us_dataset) if item["id"] == focused_id), None)
            if focused_index is None:
                # the ID does not exist or the user story could not be extracted
                if focused_id not in unextracted_ids:
                    unexistent_ids_count += 1
                continue
            
            cosine_similarities_focused = self.__calculate_cosine_similarity_row(embeddings, focused_index)
            self.__process_result_entry_focused(cosine_similarities_focused, us_dataset, focused_index, finished_indices, result)
            finished_indices.append(focused_index)

        return result, unexistent_ids_count

    def __process_result_all_pairs(self, cosine_similarities, us_dataset):
        result = []

        for i, (score_row, us_representation_1) in enumerate(zip(cosine_similarities[:-1], us_dataset[:-1])):
            for score, us_representation_2 in zip(score_row[i+1:], us_dataset[i+1:]):
                self.feed_uvl_mapper.map_similarity_result(us_representation_1, us_representation_2, score, self.threshold, result)

        return result

    def __process_result_entry_focused(self, cosine_similarities_focuesd, us_dataset, focused_index, finished_indices, result):
        focused_user_story = us_dataset[focused_index]
        for i, (score, us_representation) in enumerate(zip(cosine_similarities_focuesd, us_dataset)):
            if i == focused_index or i in finished_indices:
                continue
            self.feed_uvl_mapper.map_similarity_result(focused_user_story, us_representation, score, self.threshold, result)

    def __perform_preprocessing(self, corpus):
        if self.no_preprocessing:
            return corpus
        preprocessed_corpus = []
        for doc in corpus:
            doc_text = remove_punctuation(doc)
            if self.only_us_action:
                doc_text = get_us_action(doc_text)
            elif self.without_us_skeleton:
                doc_text = remove_us_skeleton(doc_text)
            tokens = get_tokenized_list(doc_text)
            doc_text = remove_stopwords(tokens)
            doc_text = word_stemmer(doc_text)
            doc_text = ' '.join(doc_text)
            preprocessed_corpus.append(doc_text)
        return preprocessed_corpus

    # DL semantic similarity functions

    def __setup_model(self):
        """
        Loads the DL model and tokenizer onto the device.
        """
        logging.debug(msg=f"Loading model and tokenizer for DL model: {self.modelname}")
        if self.model is None:
            module_url = configuration.get_path_for_universal_sentence_encoder()
            self.model = hub.load(module_url)

    def __create_embeddings(self, data):
        """
        Takes the model and the encodings and returns the embeddings.
        """
        return self.model(data)

    def __calculate_cosine_similarities(self, embeddings):
        """
        Takes the embeddings and creates a nxn matrix with its cosine similarities.
        """
        dimension = len(embeddings)
        cos_array = [[0]*dimension for i in range(dimension)]

        for i in range(dimension):
            for j in range(i + 1):
                cos_similarity = tf.keras.metrics.CosineSimilarity()([embeddings[i]], [embeddings[j]]).numpy()
                cos_array[i][j] = float(cos_similarity)
                if i != j: cos_array[j][i] = float(cos_similarity)
        
        return cos_array
    
    def __calculate_cosine_similarity_row(self, embeddings, index):
        """
        Takes the embeddings and an index to create a list of similarities.
        """
        _len = len(embeddings)
        cos_list = [0] * _len
        for i in range(_len):
            cos_similarity = tf.keras.metrics.CosineSimilarity()([embeddings[index]], [embeddings[i]]).numpy()
            cos_list[i] = float(cos_similarity)
        
        return cos_list
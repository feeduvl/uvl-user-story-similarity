""" techniques.wordnet module """

import logging
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from src.techniques.user_story_similarity import UserStorySimilarity
from src.techniques.preprocessing import (get_tokenized_list, pos_tagger,
    remove_punctuation, remove_stopwords,
    retrieve_corpus, get_us_action, remove_us_skeleton)
from src.feeduvl_mapper import FeedUvlMapper

class UserStorySimilarityWordnet(UserStorySimilarity):
    """
    User Story similarity analysis with WordNet and the WuPalmer Similarity
    combined with the scoring formular from mihalcea
    """

    def __init__(self, feed_uvl_mapper: FeedUvlMapper, threshold: float, without_us_skeleton: bool, only_us_action: bool) -> None:
        self.feed_uvl_mapper = feed_uvl_mapper
        self.threshold = threshold
        self.without_us_skeleton = without_us_skeleton
        self.only_us_action = only_us_action
        self.idf_of_tokens = None
        self.unique_tokens = None

    def measure_all_pairs_similarity(self, us_dataset):
        """ Similarity analysis for all pairwise user story combinations """
        corpus = retrieve_corpus(us_dataset)
        preprocessed_corpus, all_synsets, preprocessed_docs = self._perform_preprocessing(corpus)
        result = []

        if not all_synsets or len(all_synsets) == 1:
            return result

        vectorizer = TfidfVectorizer()
        self.unique_tokens = vectorizer.fit(preprocessed_docs).get_feature_names_out()
        self.idf_of_tokens = vectorizer.idf_

        for i, (us_representation_1, synsets_1, preprocessed_corpus_element_1) in enumerate(zip(us_dataset[:-1], all_synsets[:-1], preprocessed_corpus[:-1])):
            for us_representation_2, synsets_2, preprocessed_corpus_element_2 in zip(us_dataset[i+1:], all_synsets[i+1:], preprocessed_corpus[i+1:]):
                score = self._user_story_similarity(synsets_1, synsets_2, preprocessed_corpus_element_1, preprocessed_corpus_element_2)
                self.feed_uvl_mapper.map_similarity_result(us_representation_1, us_representation_2, score, self.threshold, result)

        return result

    def measure_pairwise_similarity(self, us_dataset: list, focused_ids: list[str], unextracted_ids: list[str]):
        """
        Similarity analysis for all focused user stories\n
        The user stories given in focused focused_ids are compared to every other user story in the dataset
        """
        corpus = retrieve_corpus(us_dataset)
        preprocessed_corpus, all_synsets, preprocessed_docs = self._perform_preprocessing(corpus)
        result = []
        if not all_synsets or len(all_synsets) == 1:
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
            focused_user_story = us_dataset[focused_index]
            focused_synsets = all_synsets[focused_index]
            focused_corpus_element = preprocessed_corpus[focused_index]

            for i, (us_representation_2, synsets_2, preprocessed_corpus_element_2) in enumerate(zip(us_dataset, all_synsets, preprocessed_corpus)):
                if i == focused_index or i in finished_indices:
                    continue
                score = self._user_story_similarity(focused_synsets, synsets_2, focused_corpus_element, preprocessed_corpus_element_2)
                self.feed_uvl_mapper.map_similarity_result(focused_user_story, us_representation_2, score, self.threshold, result)

            finished_indices.append(focused_index)

        return result, unexistent_ids_count

    def _user_story_similarity(self, synsets_1, synsets_2, user_story_1, user_story_2):
        if not synsets_1 or not synsets_2:
            return 0
        numerator_1, denominator_1 = self._calculate_fraction_mihalcea(synsets_1, synsets_2, user_story_1)
        numerator_2, denominator_2 = self._calculate_fraction_mihalcea(synsets_2, synsets_1, user_story_2)
        score = 0.5*(numerator_1/denominator_1 + numerator_2/denominator_2)
        return score

    def _calculate_fraction_mihalcea(self, synsets_1, synsets_2, user_story_1):
        numerator = 0.0
        denominator = 0.0
        for synset_1, word_1 in zip(synsets_1, user_story_1):
            if not synset_1:
                continue
            scores = []
            for synset_2 in synsets_2:
                if not synset_2:
                    continue
                scores.append(synset_1.wup_similarity(synset_2))
            if not scores:
                continue
            best_score = max(scores)
            idf_of_token = self._get_idf_of_token(word_1)
            best_score *= idf_of_token
            numerator += best_score
            denominator += idf_of_token
        return numerator, denominator

    def _get_idf_of_token(self, token):
        token_index = next((i for i, token in enumerate(self.unique_tokens) if token == token.lower()), None)
        if token_index is None:
            logging.warning(f'Token index for word {token} is not found. This should not occur.')
            return 0
        return self.idf_of_tokens[token_index]

    def _perform_preprocessing(self, corpus):
        preprocessed_corpus = []
        preprocessen_docs = []
        all_synsets = []
        for doc in corpus:
            # corpus preprocessing
            doc_text = remove_punctuation(doc)
            if self.only_us_action:
                doc_text = get_us_action(doc_text)
            elif self.without_us_skeleton:
                doc_text = remove_us_skeleton(doc_text)
            tokens = get_tokenized_list(doc_text)
            tokens = remove_stopwords(tokens)
            pos_tagged = pos_tagger(tokens)
            preprocessed_corpus.append(tokens)

            # predefine the synsets for each corpus element
            synsets = [self._tagged_to_synset(*tagged_word) for tagged_word in pos_tagged]
            all_synsets.append(synsets)

            preprocessen_doc_text = ' '.join(tokens)
            preprocessen_docs.append(preprocessen_doc_text)
        return preprocessed_corpus, all_synsets, preprocessen_docs

    def _tagged_to_synset(self, word, tag):
        wn_tag = self._penn_to_wn(tag)
        if wn_tag is None:
            return None

        try:
            return wn.synsets(word, wn_tag)[0]
        except Exception:
            return None

    def _penn_to_wn(self, tag):
        """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
        if tag.startswith('N'):
            return 'n'
        if tag.startswith('V'):
            return 'v'
        if tag.startswith('J'): # adjevtive
            return 'a'
        if tag.startswith('R'): # adverb
            return 'r'
        return None

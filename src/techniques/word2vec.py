from lib2to3.pgen2 import token
from src.techniques.userStorySimilarity import UserStorySimilarity
from src.techniques.preprocessing import (get_tokenized_list, pos_tagger, remove_punctuation, remove_stopwords, retrieve_corpus, word_stemmer)
from src.feedUvlMapper import FeedUvlMapper
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

class UserStorySimilarityWord2vec(UserStorySimilarity):

    def __init__(self, feed_uvl_mapper: FeedUvlMapper) -> None:
        self.feed_uvl_mapper = feed_uvl_mapper
        print("Loading word2vec model...")
        self.model: KeyedVectors = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
        print("Done loading.")

    def measure_all_pairs_similarity(self, us_dataset):
        corpus = retrieve_corpus(us_dataset)
        preprocessed_corpus, preprocessed_docs =self.perform_preprocessing(corpus)
        result = []

        vectorizer = TfidfVectorizer()
        self.unique_tokens = vectorizer.fit(preprocessed_docs).get_feature_names_out()
        self.idf_of_tokens = vectorizer.idf_

        for i, (us_representation_1, preprocessed_corpus_element_1) in enumerate(zip(us_dataset[:-1], preprocessed_corpus[:-1])):
            for us_representation_2, preprocessed_corpus_element_2 in zip(us_dataset[i+1:], preprocessed_corpus[i+1:]):
                score = self.user_story_similarity(preprocessed_corpus_element_1, preprocessed_corpus_element_2)
                self.feed_uvl_mapper.map_to_us_representation(us_representation_1, us_representation_2, score, result)

        score = self.user_story_similarity(preprocessed_corpus[0], preprocessed_corpus[3])
        
        return result

    def measure_pairwise_similarity(self, us_dataset: list, focused_ids):
        corpus = retrieve_corpus(us_dataset)
        result = []
        finished_indices = []
        # TODO: implement

       
        return result

    def user_story_similarity(self, user_story_1, user_story_2):        
        numerator_1, denominator_1 = self.calculate_term(user_story_1, user_story_2)
        numerator_2, denominator_2 = self.calculate_term(user_story_2, user_story_1)

        score = 0.5*(numerator_1/denominator_1 + numerator_2/denominator_2)
        return score

    def calculate_term(self, user_story_1, user_story_2):
        numerator = 0.0
        denominator = 0.0
        for word_1 in user_story_1:
            try:
                self.model[word_1]
            except KeyError:
                continue
            token_index = next((i for i, token in enumerate(self.unique_tokens) if token == word_1.lower()), None)
            if token_index is None:
                # TODO: consider this
                print("not found")
            idf_of_token = self.idf_of_tokens[token_index]

            best_score = 0
            scores = []
            for word_2 in user_story_2:
                try:
                    score = self.model.similarity(word_1, word_2)
                    scores.append(score)
                except KeyError:
                    continue
            best_score = max(scores)  # TODO: what if scores is empty (not very likely)

            best_score *= idf_of_token
            numerator += best_score
            denominator += idf_of_token
        return numerator, denominator

    def perform_preprocessing(self, corpus):
        preprocessed_corpus = []
        preprocessed_docs = []
        for doc in corpus:
            # corpus preprocessing
            doc_text = remove_punctuation(doc)
            tokens = get_tokenized_list(doc_text)
            tokens = remove_stopwords(tokens)
            # tokens = word_stemmer(tokens)  # TODO: is this even positively influencing the result?
            preprocessed_doc_text = ' '.join(tokens)
            preprocessed_corpus.append(tokens)
            preprocessed_docs.append(preprocessed_doc_text)

        return preprocessed_corpus, preprocessed_docs

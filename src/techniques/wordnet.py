from nltk.corpus import wordnet as wn
from src.techniques.user_story_similarity import UserStorySimilarity
from src.techniques.preprocessing import (get_tokenized_list, pos_tagger, retrieve_corpus)
from src.feeduvl_mapper import FeedUvlMapper

# TODO: ? POS tagging (tell Wordnet what POS we’re looking for)
# Wordnet only contains info on nouns, verbs, adjectives and adverbs

class UserStorySimilarityWordnet(UserStorySimilarity):

    def __init__(self, feed_uvl_mapper: FeedUvlMapper, threshold: float) -> None:
        self.feed_uvl_mapper = feed_uvl_mapper
        self.threshold = threshold

    def measure_all_pairs_similarity(self, us_dataset):
        corpus = retrieve_corpus(us_dataset)
        _, all_synsets = self.perform_preprocessing(corpus)
        result = []

        for i, (us_representation_1, synsets_1) in enumerate(zip(us_dataset[:-1], all_synsets[:-1])):
            for us_representation_2, synsets_2 in zip(us_dataset[i+1:], all_synsets[i+1:]):
                score = self.user_story_similarity(synsets_1, synsets_2)
                self.feed_uvl_mapper.map_similarity_result(us_representation_1, us_representation_2, score, self.threshold, result)

        return result

    def measure_pairwise_similarity(self, us_dataset: list, focused_ids: list[str], unextracted_ids: list[str]):
        corpus = retrieve_corpus(us_dataset)
        _, all_synsets = self.perform_preprocessing(corpus)

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
            focused_user_story = us_dataset[focused_index]
            focused_synsets = all_synsets[focused_index]
            
            for i, (us_representation_2, synsets_2) in enumerate(zip(us_dataset, all_synsets)):
                if i == focused_index or i in finished_indices:
                    continue
                score = self.user_story_similarity(focused_synsets, synsets_2)
                self.feed_uvl_mapper.map_similarity_result(focused_user_story, us_representation_2, score, self.threshold, result)
            
            finished_indices.append(focused_index)

        return result, unexistent_ids_count

    def user_story_similarity(self, synsets_1, synsets_2):
        score, count = 0.0, 0
        for synset in synsets_1:
            best_score = max([synset.wup_similarity(ss) for ss in synsets_2])  # TODO: use WuP sim
            if best_score is not None:
                score += best_score
                count += 1
        score /= count  # TODO: handle zero division
        return score

    def perform_preprocessing(self, corpus):
        preprocessed_corpus = []
        all_synsets = []
        for doc in corpus:
            # corpus preprocessing
            tokens = get_tokenized_list(doc)
            doc_text = pos_tagger(tokens)
            preprocessed_corpus.append(doc_text)

            # predefine the synsets for each corpus element
            synsets = [self.tagged_to_synset(*tagged_word) for tagged_word in doc_text]
            synsets = [ss for ss in synsets if ss]
            all_synsets.append(synsets)
        return preprocessed_corpus, all_synsets

    def tagged_to_synset(self, word, tag):
        wn_tag = self.penn_to_wn(tag)
        if wn_tag is None:
            return None
    
        try:
            return wn.synsets(word, wn_tag)[0]
        except:
            return None

    def penn_to_wn(self, tag):
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

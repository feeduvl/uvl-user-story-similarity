""" techniques.user_story_similarity module """

from abc import ABC, abstractmethod

class UserStorySimilarity(ABC):
    """ Abstract class which specifies the methods the respective NLP techniques have to implement """

    @abstractmethod
    def measure_all_pairs_similarity(self, us_dataset: list):
        """ Abstract method for similarity analysis of all pairwise combinations """

    @abstractmethod
    def measure_pairwise_similarity(self, us_dataset: list, focused_ids: list[str], unextracted_ids: list[str]):
        """ Abstract method for similarity analysis of all focused pairwise combinations """

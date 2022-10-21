from abc import ABC, abstractmethod

class UserStorySimilarity(ABC):

    @abstractmethod
    def measure_all_pairs_similarity(self):
        pass

    @abstractmethod
    def measure_pairwise_similarity(self):
        pass

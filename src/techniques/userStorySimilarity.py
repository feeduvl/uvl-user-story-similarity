from abc import ABC, abstractmethod

class UserStorySimilarity(ABC):
    @abstractmethod
    def measure_similarity(self):
        pass

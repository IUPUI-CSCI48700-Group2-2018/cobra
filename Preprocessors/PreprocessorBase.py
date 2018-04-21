from abc import ABC, abstractmethod

class PreprocessorBase(ABC):

    @abstractmethod
    def preprocess(self, data):
        pass

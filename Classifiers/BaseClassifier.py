from abc import ABC, abstractmethod

class BaseClassifier(ABC):

    @abstractmethod
    def fit(dataGenerator):
        pass

    def predict(data):
        pass

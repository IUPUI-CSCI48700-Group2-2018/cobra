from abc import ABC, abstractmethod

class BaseClassifier(ABC):

    @abstractmethod
    def fit(data, validation):
        pass

    def predict(data):
        pass

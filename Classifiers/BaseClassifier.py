from abc import ABC, abstractmethod

class BaseClassifier(ABC):

    @abstractmethod
    def fit(data, labels):
        pass

    def predict(data):
        pass

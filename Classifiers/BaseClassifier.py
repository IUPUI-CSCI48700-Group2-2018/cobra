from abc import ABC, abstractmethod

class BaseClassifier(ABC):

    @abstractmethod
    def fit(train, test=None):
        pass

    def predict(data):
        pass

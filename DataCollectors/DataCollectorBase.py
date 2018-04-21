from abc import ABC, abstractmethod

class DataCollectorBase(ABC):
    @abstractmethod
    def collectData(self):
        pass

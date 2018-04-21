from .DataCollectorBase import DataCollectorBase
from keras.preprocessing.image import ImageDataGenerator

class LocalDataCollector(DataCollectorBase):

    def collectData(self):
        train_batches = ImageDataGenerator().flow_from_directory(
            'data/CarsID/train', target_size=(224, 224), classes=['c', 'm'], batch_size=10)  #c stand for Camry
        valid_batches = ImageDataGenerator().flow_from_directory(
            'data/CarsID/valid', target_size=(224, 224), classes=['c', 'm'], batch_size=10)  #m stand for Mercedes
        test_batches = ImageDataGenerator().flow_from_directory(
            'data/CarsID/test', target_size=(224, 224), classes=['c', 'm'], batch_size=4)

        return train_batches, test_batches, valid_batches

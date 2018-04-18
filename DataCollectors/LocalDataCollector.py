from .DataCollectorBase import DataCollectorBase
from .DataCollection import DataCollection
from keras.preprocessing.image import ImageDataGenerator

class LocalDataCollector(DataCollectorBase):

    def collectData(self):
        train_batches = ImageDataGenerator().flow_from_directory(
            'CarsID/train', target_size=(224, 224), classes=['c', 'm'], batch_size=10)  #c stand for Camry
        valid_batches = ImageDataGenerator().flow_from_directory(
            'CarsID/valid', target_size=(224, 224), classes=['c', 'm'], batch_size=10)  #m stand for Mercedes
        test_batches = ImageDataGenerator().flow_from_directory(
            'CarsID/test', target_size=(224, 224), classes=['c', 'm'], batch_size=4)

        return DataCollection(train_batches, test_batches,valid_batches)

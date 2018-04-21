from .DataCollectorBase import DataCollectorBase
from keras.preprocessing.image import ImageDataGenerator

class LocalDataCollector(DataCollectorBase):

    def collectData(self,directory):
        data = ImageDataGenerator().flow_from_directory(
            directory,
            target_size=(224, 224),
            batch_size=10)


        # valid_batches = ImageDataGenerator().flow_from_directory(
        #     'CarsID/valid', target_size=(224, 224), classes=['c', 'm'], batch_size=10)  #m stand for Mercedes
        # test_batches = ImageDataGenerator().flow_from_directory(
        #     'CarsID/test', target_size=(224, 224), classes=['c', 'm'], batch_size=4)

        return data

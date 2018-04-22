from .DataCollectorBase import DataCollectorBase
from keras.preprocessing.image import ImageDataGenerator

class LocalDataCollector(DataCollectorBase):

    def collectData(self,directory):
        dataGenerator = ImageDataGenerator(validation_split=0.3)

        train = dataGenerator.flow_from_directory(
            "data/CobraCleaned",
            target_size=(224, 224),
            subset="training",
            # color_mode='grayscale',
            batch_size=32)

        test = dataGenerator.flow_from_directory(
            "data/CobraCleaned",
            target_size=(224, 224),
            subset="validation",
            # color_mode='grayscale',
            batch_size=32)

        return train, test

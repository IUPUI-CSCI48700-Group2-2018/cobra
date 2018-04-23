from .DataCollectorBase import DataCollectorBase
from keras.preprocessing.image import ImageDataGenerator

class LocalDataCollector(DataCollectorBase):

    def __init__(self, directory, preprocessor=None):
        self.directory = directory
        self.preprocessor = preprocessor

    def collectData(self):
        dataGenerator = ImageDataGenerator(validation_split=0.3,
            preprocessing_function=self.preprocessor)

        train = dataGenerator.flow_from_directory(
            self.directory,
            target_size=(224, 224),
            subset="training",
            batch_size=32)

        test = dataGenerator.flow_from_directory(
            self.directory,
            target_size=(224, 224),
            subset="validation",
            batch_size=32)

        return train, test

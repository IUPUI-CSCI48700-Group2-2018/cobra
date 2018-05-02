from .DataCollectorBase import DataCollectorBase
from keras.preprocessing.image import ImageDataGenerator

class LocalDataCollector(DataCollectorBase):

    def __init__(self, directory, preprocessor=None):
        self.directory = directory
        self.preprocessor = preprocessor

    def collectData(self):
        dataGenerator = ImageDataGenerator(preprocessing_function=self.preprocessor)

        train = dataGenerator.flow_from_directory(
            self.directory+"/train",
            target_size=(224, 224),
            batch_size=32)

        test = dataGenerator.flow_from_directory(
            self.directory+"/test",
            target_size=(224, 224),
            batch_size=2000,
            shuffle=False)

        return train, test

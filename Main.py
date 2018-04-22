from Classifiers import Vgg16Classifier
from DataCollectors import LocalDataCollector

from Preprocessors import PreprocessorImpl
from Application import Application

dataCollector = LocalDataCollector()
preprocessor = PreprocessorImpl()
classifier = Vgg16Classifier()

app = Application(dataCollector, preprocessor, classifier)

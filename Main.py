from Classifiers import Vgg16Classifier
from DataCollectors import LocalDataCollector
from preprocessing import simplePreprocessing
from Application import Application

dataCollector = LocalDataCollector("data/CobraCleaned",simplePreprocessing)
classifier = Vgg16Classifier("testModel")

app = Application(dataCollector, classifier)
app.run()

from Classifiers import Vgg16Classifier
from DataCollectors import LocalDataCollector
from preprocessing import simplePreprocessing
from Application import Application

dataCollector = LocalDataCollector("data/CobraCleanedDiv")
classifier = Vgg16Classifier("dummy")

app = Application(dataCollector, classifier)
app.run()

from Classifiers import Vgg16Classifier
from DataCollectors import LocalDataCollector
from preprocessing import simplePreprocessing
from Application import Application

dataCollector = LocalDataCollector("data/CobraCleanedDiv",simplePreprocessing)
classifier = Vgg16Classifier("modelSobelPreprocessing")

app = Application(dataCollector, classifier)
app.run()

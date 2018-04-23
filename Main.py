from Classifiers import Vgg16Classifier
from DataCollectors import LocalDataCollector
from preprocessing import simplePreprocessing
from Application import Application

dataCollector = LocalDataCollector("data/CobraSmall",simplePreprocessing)
classifier = Vgg16Classifier()

app = Application(dataCollector, classifier)
app.run()

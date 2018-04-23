import numpy as np
from sklearn.metrics import f1_score

class Application:
    def __init__(self, dataCollector, classifier):
        self.dataCollector = dataCollector
        self.classifier = classifier

    def run(self):
        train,test = self.dataCollector.collectData()

        self.classifier.fit(train)
        # prediction = self.classifier.predict(test);
        #
        # _, testLabels = next(test)
        # print(np.argmax(testLabels,axis=1))
        # print(prediction)

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class Application:
    def __init__(self, dataCollector, classifier):
        self.dataCollector = dataCollector
        self.classifier = classifier

    def run(self):
        train,test = self.dataCollector.collectData()

        # self.classifier.fit(train)

        print("Reading In Test Data...")
        testData, testLabels = next(test)
        prediction = self.classifier.predict(testData);

        testLabels = np.argmax(testLabels,axis=1)
        print(testLabels)
        print(prediction)
        print(f1_score(testLabels,prediction, average="macro"))
        print(accuracy_score(testLabels,prediction))
        return prediction, testLabels

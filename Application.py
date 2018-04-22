import numpy as np
from sklearn.metrics import f1_score

class Application:
    def __init__(self, dataCollector, preprocessor, classifier):
        train,test = dataCollector.collectData("data/CobraSmall/train")
        # processedData = preprocessor.preprocess(trainData)
        classifier.fit(train,test)

        # Test
        # testData = dataCollector.collectData("data/CobraSmall/test")
        # prediction = classifier.predict(testData);
        #
        # _, testLabels = next(testData)
        # print(np.argmax(testLabels,axis=1))
        # print(prediction)
        # print()

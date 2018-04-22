import numpy as np
from sklearn.metrics import f1_score

class Application:
    def __init__(self, dataCollector, preprocessor, classifier):
        train,test = dataCollector.collectData("data/CobraSmall/train")
        # processedData = preprocessor.preprocess(trainData)
        classifier.fit(train)

        prediction = classifier.predict(test);

        _, testLabels = next(test)
        print(np.argmax(testLabels,axis=1))
        print(prediction)

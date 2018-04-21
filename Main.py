from Classifiers import Vgg16Classifier
from DataCollectors import LocalDataCollector

dataCollector = LocalDataCollector()
train, test, validation = dataCollector.collectData()

c = Vgg16Classifier()
c.fit(train,validation)
pred = c.predict(test)
print(pred)

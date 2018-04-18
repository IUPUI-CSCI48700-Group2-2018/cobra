from Classifiers import Vgg16Classifier
from DataCollectors import LocalDataCollector

dataCollector = LocalDataCollector()
data = dataCollector.collectData()

c = Vgg16Classifier()
c.fit(data.train,data.validation)
pred = c.predict(data.test)
print(pred)

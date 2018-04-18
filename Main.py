from Classifiers import vgg16Classifier
from keras.preprocessing.image import ImageDataGenerator

train_batches = ImageDataGenerator().flow_from_directory(
    'CarsID/train', target_size=(224, 224), classes=['c', 'm'], batch_size=10)  #c stand for Camry
valid_batches = ImageDataGenerator().flow_from_directory(
    'CarsID/valid', target_size=(224, 224), classes=['c', 'm'], batch_size=10)  #m stand for Mercedes
test_batches = ImageDataGenerator().flow_from_directory(
    'CarsID/test', target_size=(224, 224), classes=['c', 'm'], batch_size=4)

c = vgg16Classifier()
c.fit(train_batches,valid_batches)
pred = c.predict(test_batches)
print(pred)

from .BaseClassifier import BaseClassifier
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

class Vgg16Classifier(BaseClassifier):

    # Convert the vgg16 model to a sequential model, remove the last
    # layer, disable training on each layer of the sequential model, and
    # add a new layer
    def fit(self, dataGenerator):
        vgg16Model = VGG16()
        model = Sequential()
        for layer in vgg16Model.layers:
            model.add(layer)
            
        model.layers.pop()
        for layer in model.layers:
            layer.trainable = False

        model.add(Dense(2, activation='softmax'))

        model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit_generator(train, steps_per_epoch=4, validation_data=validation, validation_steps=4,
                                 epochs=5, verbose=2)

        self.model = model

    def predict(self, test):
        return np.argmax(self.model.predict_generator(test, steps=1, verbose=0),axis=1)

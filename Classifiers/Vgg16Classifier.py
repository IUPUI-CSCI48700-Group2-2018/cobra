from .BaseClassifier import BaseClassifier
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import time

class Vgg16Classifier(BaseClassifier):

    # Convert the vgg16 model to a sequential model, remove the last
    # layer, disable training on each layer of the sequential model, and
    # add a new layer with the number of nodes equaling the number of classes
    def fit(self, train, test=None):
        vgg16Model = VGG16()
        model = Sequential()
        for layer in vgg16Model.layers:
            model.add(layer)

        model.layers.pop()
        for layer in model.layers:
            layer.trainable = False

        model.add(Dense(len(train.class_indices), activation='softmax'))
        model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit_generator(train, validation_data=test, epochs=40,verbose=1)
        model.save_weights('model.h5')

        self.model = model

    def predict(self, data):
        return np.argmax(self.model.predict_generator(data, verbose=1),axis=1)

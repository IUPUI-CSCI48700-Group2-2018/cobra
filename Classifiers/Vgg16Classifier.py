from .BaseClassifier import BaseClassifier
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import time

class Vgg16Classifier(BaseClassifier):

    def __init__(self, modelName):
        self.modelName = modelName

    # Convert the vgg16 model to a sequential model, remove the last
    # layer, disable training on each layer of the sequential model, and
    # add a new layer with the number of nodes equaling the number of classes
    def fit(self, train, test=None):
        vgg16Model = VGG16()

        vgg16Model.layers.pop()

        for layer in vgg16Model.layers:
            layer.trainable = False

        model = vgg16Model.output

        model = Dense(len(train.class_indices), activation='softmax')(model)
        model = Model(inputs=vgg16Model.input, outputs=model)
        model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit_generator(train, validation_data=test, epochs=50,verbose=1)
        model.save(self.modelName+".h5")

        self.model = model

    def predict(self, data):
        return np.argmax(self.model.predict_generator(data, verbose=1),axis=1)

from .BaseClassifier import BaseClassifier
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

class Vgg16Classifier(BaseClassifier):

    def fit(self, train, validation):
        vgg16_model = VGG16()
        Seq_modelS = Sequential()          #create linear layers to modify vgg16
        for layers in vgg16_model.layers:  # loop in each layer in vigg16
            Seq_modelS.add(layers)         #and add each layer to the sequentail model

        Seq_modelS.layers.pop()            #pop the last layer,prediction (Dense)(None,1000)
                                           #becuse this model can analysis 1000 differnt categories

        for layer in Seq_modelS.layers:    # for now there is only to categories
            layer.trainable = False        # dont trian the layers because they have been trained in vgg16

        Seq_modelS.add(Dense(2, activation='softmax'))# add this layer to the end of model,becuse we only have 2 categories to train for now

        Seq_modelS.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        Seq_modelS.fit_generator(train, steps_per_epoch=4, validation_data=validation, validation_steps=4,
                                 epochs=1, verbose=2)

        self.model = Seq_modelS

    def predict(self, test):
        return np.argmax(self.model.predict_generator(test, steps=1, verbose=0),axis=1)

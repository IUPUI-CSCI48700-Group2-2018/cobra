from .BaseClassifier import BaseClassifier
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
import numpy as np

class ConvolutionClassifier(BaseClassifier):

    def fit(self, train, test=None):
        model = Sequential()

        model.add(Convolution2D(16, (3, 3), border_mode='same', input_shape=(64, 64, 1), activation='relu'))
        model.add(Convolution2D(16, (3, 3), border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))

        model.add(Convolution2D(32, (3, 3), border_mode='same', activation='relu'))
        model.add(Convolution2D(32, (3, 3), border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))

        model.add(Convolution2D(64, (3, 3), border_mode='same', activation='relu'))
        model.add(Convolution2D(64, (3, 3), border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))

        model.add(Convolution2D(128, (3, 3), border_mode='same', activation='relu'))
        model.add(Convolution2D(128, (3, 3), border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(len(train.class_indices), activation='softmax'))
        model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit_generator(train, validation_data=test, epochs=1,verbose=1)

        self.model = model

    def predict(self, data):
        return np.argmax(self.model.predict_generator(data, steps=1, verbose=0),axis=1)

#**************************************************************************************/
#* Name:      
#* Purpose:   
#* Author:   Alsaadi Y,Amos Enderson,Christopher Ash,Elias Kraihanzel,Lincoln Anderson
#* Created:  04-03-2018
#* Copyright: Alsaadi Y,Amos Enderson,Christopher Ash,Elias Kraihanzel,Lincoln Anderson
#* License:
#***************************************************************************************

import numpy as np
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy,categorical_crossentropy  
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
import itertools
import matplotlib.pyplot as plt
import  PIL

train_batches = ImageDataGenerator().flow_from_directory(
    'CarsID/train', target_size=(224, 224), classes=['c', 'm'], batch_size=10)  #c stand for Camry
valid_batches = ImageDataGenerator().flow_from_directory(
    'CarsID/valid', target_size=(224, 224), classes=['c', 'm'], batch_size=10)  #m stand for Mercedes
test_batches = ImageDataGenerator().flow_from_directory(
    'CarsID/test', target_size=(224, 224), classes=['c', 'm'], batch_size=4)


test_image2, test_labels2 = next(test_batches)  #to label the classes
print(test_labels2)                            
print()

vgg16_model = keras.applications.vgg16.VGG16()
#vgg16_model.summary()             #To print the 16 layers summary of vigg16 model   

Seq_modelS = Sequential()          #create linear layers to modify vgg16
for layers in vgg16_model.layers:  # loop in each layer in vigg16
    Seq_modelS.add(layers)         #and add each layer to the sequentail model

Seq_modelS.layers.pop()            #pop the last layer,prediction (Dense)(None,1000)
                                   #becuse this model can analysis 1000 differnt categories

for layer in Seq_modelS.layers:    # for now there is only to categories 
    layer.trainable = False        # dont trian the layers because they have been trained in vgg16  
    
Seq_modelS.add(Dense(2, activation='softmax'))# add this layer to the end of model,becuse we only have 2 categories to train for now
#Seq_modelS.summary()                         #print summary of the modified model



Seq_modelS.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
Seq_modelS.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4,
                         epochs=5, verbose=2)

predictions = Seq_modelS.predict_generator(test_batches, steps=1, verbose=0)
print(predictions)
print()
print(test_batches.class_indices)         #print how the classes is labeled in keras
print()
print(np.argmax(predictions, axis=1))     #print the final result by labels 


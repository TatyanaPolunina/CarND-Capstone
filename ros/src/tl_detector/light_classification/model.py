import os
import csv
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Lambda, Flatten, Dense, Conv2D, Activation
import tensorflow as tf
from sklearn.utils import shuffle
import yaml

folder = '/mnt/shared/sim_training_data/'
model_name = "sim_model.h5"
yaml_name = 'sim_data_annotations.yaml'

#read the data info from csv file

def get_class(class_str):
    if (class_str == 'Green'):
        return 2;
    if (class_str == 'Yellow'):
        return 1;
    if (class_str == 'Red'):
        return 0;
    return 0;
    
def read_the_data():
    samples = []
    with open(folder+yaml_name, 'r') as stream:
        try:
            data = yaml.safe_load(stream)      
        except yaml.YAMLError as exc:
            print(exc)
    for image in data:
       for b in image['annotations']:
              sample = dict();
              sample['filename'] = folder + image['filename']
              if b['x_width'] < 5 or b['y_height'] < 5:
                continue;
              border = [max(0, int(b['xmin'])), int(b['xmin'] + b['x_width']), max(0, int(b['ymin'])), int(b['ymin'] + b['y_height'])]
              sample['border'] = border
              sample['class'] = get_class(b['class'])
              samples.append(sample)
              
    shuffle(samples)
    #split the data to train (80%) and validation (20%) set
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples
    
labels = [[1,0,0], [0,1,0], [0,0,1]]

#load batch samples data generaton
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            lights = []
            correction = 0.2
            for batch_sample in batch_samples:
                img_name = batch_sample['filename']
                traffic_light = batch_sample['class']
                image = cv2.imread(img_name) 
                brd = batch_sample['border']
                if (brd[3] - brd[2] == 0 or brd[1] - brd[0] == 0):
                    print(brd)
                try:
                    final_image = cv2.resize(image[brd[2]:brd[3], brd[0]:brd[1]], (48, 96));
                except Exception:
                    print(batch_sample)
                images.append(final_image);                       
                lights.append(labels[traffic_light])
               
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(lights)
            yield shuffle(X_train, y_train)

def get_model_architecture():
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(96, 48, 3),\
            output_shape=(96, 48, 3)))
    
    #provide convolution layers
    model.add(Conv2D(3, (3, 3), strides=(2, 1), padding='same', input_shape=(48, 48, 3)))
    #add VGG pretrained model
    vgg = VGG16(weights='imagenet', include_top=False,\
                       input_shape=(48, 48, 3))    
    #use already pretrained parameters for current model
    for layer in vgg.layers: 
        vgg.trainable = False
    model.add(vgg)   
    #add fully connected layer
    model.add(Flatten())
    model.add(Dense(256))
    #add layer to receive needed output
    model.add(Dense(3))    
    model.add(Activation('softmax'))
    #compile model with adam optimized and mse error.
    model.compile('adam', loss='categorical_crossentropy')
    return model
                    
batch_size=32
# compile and train the model using the generator function
train_samples, validation_samples = read_the_data()
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
model = get_model_architecture()
#see the final model architecture
print(model.summary())
model.fit_generator(train_generator, \
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=np.ceil(len(validation_samples)/batch_size), \
            epochs=5, verbose=1)
model.save('model.h5')

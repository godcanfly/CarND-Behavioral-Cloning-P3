__author__ = 'zhiyong_wang'
import csv
import cv2
import numpy as np
import os

def update_with_path(filename_path,file_directory):
    filename = filename_path.split('/')[-1]
    current_path = file_directory + '/IMG/' + filename
    return current_path





def capture_data_from_folder(data_path,camera_correction = 0.2):

    lines = []
    images = []
    measurements = []
    with open(data_path + '/driving_log.csv') as csvfile:
        next(csvfile)#skip first line
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        for i in range(0,3):
            filename = line[i]
            image_path = update_with_path(filename,data_path)
            image = cv2.imread(image_path)
            measurement = float(line[3])
            if i ==1:
                measurement = measurement + camera_correction

            if i == 2:
                measurement = measurement - camera_correction

            images.append(image)
            measurements.append(measurement)
            images.append(np.fliplr(image))
            measurements.append(-measurement)
    return images,measurements


def capture_data_from_dir(data_dir):
    images = []
    measurements = []
    for name in os.listdir(data_dir):
        full_path = os.path.join(data_dir, name)
        if os.path.isdir(full_path):
            print("full path is ",full_path)
            sub_images,sub_measurements = capture_data_from_folder(full_path)
            images.extend(sub_images)
            measurements.extend(sub_measurements)
    return images,measurements


images, measurements = capture_data_from_dir('../../beclonedata/')



X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda  x: x/255.0 - 0.5,input_shape=(160,320,3) ))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))


model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=1)
model.save('model.h5')
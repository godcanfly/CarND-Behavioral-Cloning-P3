__author__ = 'zhiyong_wang'
import csv
import cv2
import numpy as np
import os
from utility import process_image_file


use_generator = True


DATA_SOURCE_DIRECTORY ='../../beclonedata/'
CAMERA_CORRECTION = 0.2

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
            image = process_image_file(image_path)
            if image is None:
                continue
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


def get_all_lines_from_dir(data_dir):
    lines = []
    for name in os.listdir(data_dir):
        full_path = os.path.join(data_dir, name)
        if os.path.isdir(full_path):
            print("full path is ",full_path)
            with open(full_path + '/driving_log.csv') as csvfile:
                next(csvfile)#skip first line
                reader = csv.reader(csvfile)
                for line in reader:
                    line.append(name)
                    lines.append(line)

    return lines

def generator(samples, batch_size=32):

    from random import shuffle
    import sklearn
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:

                for i in range(0,3):
                    filename = batch_sample[i]
                    image_path = update_with_path(filename,DATA_SOURCE_DIRECTORY + batch_sample[-1])
                    image = process_image_file(image_path)
                    if image is None:
                        continue
                    measurement = float(batch_sample[3])
                    if i ==1:
                        measurement = measurement + CAMERA_CORRECTION

                    if i == 2:
                        measurement = measurement - CAMERA_CORRECTION

                    images.append(image)
                    measurements.append(measurement)
                    images.append(np.fliplr(image))
                    measurements.append(-measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


# Try small set of data
#images, measurements = capture_data_from_folder('../../beclonedata/center')


from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda  x: x/255.0 - 0.5,input_shape=(100,320,1) ))


model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))


model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')


if use_generator:

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(get_all_lines_from_dir(DATA_SOURCE_DIRECTORY), test_size=0.2)


    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples), validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=1)
else:


    images, measurements = capture_data_from_dir(DATA_SOURCE_DIRECTORY)



    X_train = np.array(images)
    y_train = np.array(measurements)

    model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=1)

model.save('model.h5')
import numpy as np
import matplotlib as plt
import csv
import pickle
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense


# path_data = '../windows-sim/windows_sim/data'
path_data = '../windows-sim'
lines = []
images = []
measurements = []

def load_data():
    print('Load Data')
    with open(path_data) as csvfile:
        reader = csv.reader()
        for line in reader :
            lines.append(line)

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        path_image = path_data + '/IMG' + filename
        image = cv2.imread(path_image)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
    return


load_data()
X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch= 7)

model.save('model.h5')


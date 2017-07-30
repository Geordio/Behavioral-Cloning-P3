import csv

import scipy.misc
import cv2
import random
import numpy as np

from keras.layers import Flatten, Dense, Cropping2D, Convolution2D, BatchNormalization, Lambda, Activation, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle
from helpers import get_angles_only, plot_angles, plot_figure, process_batch_no_images

path_data = '../../windows-sim/windows_sim/data'
path_data_new = '../../windows-sim/windows_sim/new_data'
filename_csv = 'driving_log.csv'
#path_data = '../windows-sim'
lines = []
images = []
image_paths_centre = []
image_paths_right = []
image_paths_left = []

measurements = []
centre_col = 0
left_col = 1
right_col = 2
angle_col = 3
angle_offset = 0.19
straight_thres = 0.1


# orginal image is 160 x 320. I modified to 70 x 320
new_width = 200
new_height = 66



def flip_left_right(image):
    return cv2.flip(image, 1 )

# method to adjust the brightness of the image.
# creates a random number between 0 and 1 to decide if the brightness should be adjusted.
#  converts to HSV colour sace, then adjusts the V by random value
#  between 0.5 and 1.
def adjust_brightness(image):
    # apply a random brightness adjustment
    adjustFlag = random.randint(0,1)
    if adjustFlag ==1:
        # convert image to hsv colour space
        out_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        adjustment_factor = np.random.uniform(0.5, 1)
        # adjust the v (brightness)
        out_image[:,:,2] = out_image[:,:,2]*adjustment_factor
        # convert it back to rgb colour space
        out_image = cv2.cvtColor(out_image,cv2.COLOR_HSV2RGB)

        return out_image
    else:
#        print('no brightness adjust')
        return image



# for each line in the dataset
# extract the image paths and steering angles
# for the side cameras, calculate an angle to compensate for the offset position
# return an array array containing angles and image paths for each camera.
def organise_dataset(dataset):
 #   i = 0
    centre_angles = []
    centre_paths = []

    left_angles = []
    left_paths = []
    right_angles = []
    right_paths = []

    for sample in dataset:
        angle, centre, left, right = extract_data(sample)

        left_image_angle = angle + angle_offset
        right_image_angle = angle - angle_offset
        # print('angle: {} {} {}'.format(angle,left_image_angle,right_image_angle ))

        centre_angles.append(angle)
        centre_paths.append(centre)
        left_angles.append(left_image_angle)
        left_paths.append(left)
        right_angles.append(right_image_angle)
        right_paths.append(right)

# return the angles and image paths
    return centre_angles, centre_paths, left_angles, left_paths, right_angles, right_paths

#
# partitions the data into 3 categories, left right and centre
# parses the data from teh left, right and centre images, and
# groups into either centre or non centre classifications.
# also balances dataset
def partition_data(centre_angles, centre_paths, left_angles, left_paths, right_angles, right_paths):
    #   i = 0
    central_angles = []
    central_images = []
    non_central_angles = []
    non_central_images = []

    print('centre_angles len: {}' .format(len(centre_angles)))
    print('centre_paths len: {}' .format(len(centre_paths)))
    print('left_angles len: {}' .format(len(left_angles)))
    print('left_paths len: {}' .format(len(left_paths)))
    print('right_angles len: {}' .format(len(right_angles)))
    print('right_paths len: {}' .format(len(right_paths)))

    overall_angles =centre_angles + left_angles + right_angles
    plot_angles(overall_angles)
    # plot_angles(left_angles)
    # plot_angles(right_angles)
    overall_images =centre_paths + left_paths + right_paths
    for angle, image in zip(overall_angles, overall_images):
        # if its a steering angle close to staright ahead
        if (abs(angle) < 0.05):
            # randomly discard half the central images
            dropNotDrop = random.randint(0, 1)
            if dropNotDrop == 0:
                central_angles.append(angle)
                central_images.append(image)
        else :
            # if its non a straight-ish steering angle
            if (abs(angle) < angle_offset + 0.02 ) & (abs(angle) > angle_offset - 0.02):
                # try to flatten out some of the peaks caused by use of the side cameras.
                dropNotDrop = random.randint(0, 2)
                if dropNotDrop == 0:
                    non_central_angles.append(angle)
                    non_central_images.append(image)
            else:
                non_central_angles.append(angle)
                non_central_images.append(image)
    plot_angles(non_central_angles)
    plot_angles(central_angles)
    return central_angles + non_central_angles, central_images + non_central_images
    # return central_angles, central_images, non_central_angles, non_central_images

# get samples from driving data csv
# return a list of lines from the csv
def get_driving_data(path_data):
    path_full = path_data+ '/'+ filename_csv
    with open(path_full) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader :
            lines.append(line)

    return lines

# method to extract the image paths and angles from a line in the csv
# returns angle, path_centre, path_left, path_right
def extract_data(line):
    path_centre = line[centre_col]
    path_left = line[left_col]
    path_right = line[right_col]
    angle = float(line[angle_col])

    # print('sample: {}, {}, {}, {}, {}' .format(path_centre))
    return angle, path_centre, path_left, path_right




def resize_normalise(image):
    image = image[66:-24, :]
    image = (scipy.misc.imresize(image, [66, 200]) / 255.0)
    return image



def generator_train_all(images_files, angles, batch_size):
#    print('running generator')
    num_samples = len(images_files)
    while True:
        angles, images_files = shuffle(angles, images_files)

        for offset in range(0, num_samples, batch_size):
            # print('offset: {}'.format(offset))

            batch_image_files = images_files[offset:offset+batch_size]
            batch_angles = angles[offset:offset+batch_size]
            out_images = []
            out_angles = []
            for image_file, angle in zip(batch_image_files,batch_angles):
                # print('image and angle: {}, {}'.format(image_file, angle))
                image = cv2.imread(image_file)
                image = resize_normalise(image)

                flipNotFlip = random.randint(0, 1)
                # don't flip the image
                if flipNotFlip > 0:
                    out_angles.append(angle)
                    out_images.append(image)
                # flip the image, and the angle
                else:
                    out_images.append(flip_left_right(image))
                    out_angles.append(-angle)

            yield shuffle(np.array(out_images), np.array(out_angles))

# generator to feed training data
# randomly selects samples
def generator_train_rnd(images_files, angles, batch_size):
    num_samples = len(images_files)


    while True:
        # shuffle the dataset (which is in 2 arrays, central and non central
        angles, images_files = shuffle(angles, images_files)
        out_images = []
        out_angles = []
        for i in range(batch_size):
            rnd = random.randint(0, num_samples-1)
            image = cv2.imread(images_files[rnd])
            image = resize_normalise(image)

            flipNotFlip = random.randint(0, 1)
            # don't flip the image
            if flipNotFlip > 0:
                out_angles.append(angles[rnd])
                out_images.append(image)
            # flip the image, and the angle
            else:
                out_images.append(flip_left_right(image))
                out_angles.append(-angles[rnd])

        yield shuffle(np.array(out_images), np.array(out_angles))

# helper function for playing around and displaying augmentations
def display_augment(samples):
    for batch_sample in samples:
        path, angle = extract_data(batch_sample)
        cv2.imshow('orig',cv2.imread(path))
        center_image = adjust_brightness(cv2.imread(path))
        path_left_image = batch_sample[left_col]
        path_right_image = batch_sample[right_col]
        left_image = cv2.imread(path_left_image)
        right_image = cv2.imread(path_right_image)

# generator for teh validation step.
# does not do anything other than provide batches
# only uses the centre images
def generator_valid( central_images_file, central_angles, batch_size):

    num_samples_centre = len(central_images_file)

    while True:
        # shuffle the set
        central_angles, central_images_file = shuffle(central_angles, central_images_file)

        images = []
        angles = []
        for i in range(batch_size):
            rnd = random.randint(0, num_samples_centre - 1)

            central_images = cv2.imread(central_images_file[rnd])
            central_images = resize_normalise(central_images)
            angles.append(central_angles[rnd])
            images.append(central_images)

        yield shuffle(np.array(images), np.array(angles))



#load_data
# there are 2 datasets from training.
samples = get_driving_data(path_data)
samples_new = get_driving_data(path_data_new)
# join the 2 datasets
samples = samples+samples_new


# organise the data into centre, left and right camera data
centre_angles, centre_paths, left_angles, left_paths, right_angles, right_paths =organise_dataset(samples)
# plot_angles(centre_angles)
# plot_angles(left_angles)
# plot_angles(right_angles)
balanced_angles, balanced_images =  partition_data(centre_angles, centre_paths, left_angles, left_paths, right_angles, right_paths)
#plot_angles( balanced_angles)

#shuffle the samples prior to splitting the dataset
balanced_images, balanced_angles = shuffle(balanced_images, balanced_angles)

images_train, images_valid, angles_train,  angles_valid = train_test_split(balanced_images, balanced_angles,test_size=0.2, random_state=48)

print('images_train total {}'.format(len(angles_train)))

# set up the generators
train_generator = generator_train_all(images_train, angles_train, batch_size=128)
validation_generator = generator_valid(images_valid,angles_valid, batch_size=128)

###model definition
model = Sequential()

#model.add(Lambda(lambda x: x/255-.05 , input_shape=(160,320,3)))
#There is an issue with Keras Lambda above, that means that it won't save the model.
#used the work around below...
#https://stackoverflow.com/questions/41847376/keras-model-to-json-error-rawunicodeescape-codec-cant-decode-bytes-in-posi


# implement convnet as per the nvidia white paper on end to end learning
model.add(Convolution2D(24,5,5,subsample= (2,2), input_shape=(66, 200, 3),activation = 'relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5,subsample= (2,2), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5,subsample= (2,2), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,subsample= (1,1), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,subsample= (1,1), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# compile model. Use Adam Optimiser
model.compile(loss = 'mse', optimizer=Adam())

# run the fit generator
samples_to_include = int(len(images_train)/2)

model.fit_generator(train_generator, samples_per_epoch=samples_to_include,\
                    validation_data=validation_generator, nb_val_samples=len(angles_valid), nb_epoch=5)


# print the model summary
print(model.summary())
# save the model
model.save('../model.h5')

import csv
import cv2

import sklearn
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda

############################## Image process support methods #####################
# Method to process image for random brightness
def randomBrightnessImage(image):
    # Convert to HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = np.array(image, dtype=np.float64)
    # Splitting to H, S, V, to work on V value for brightness
    h, s, v = cv2.split(image)
    # Calculating correction factor with random number
    correction = np.random.uniform() + 0.3
    v *= correction
    image = cv2.merge([h, s, v])
    image = np.array(image, dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

# Method to add random polygon shape shadows to original image
# Polygon makes it little close to real road simulation, since
# on road we see all random shape images
def shadowPolygon(im):
    # Adding alpha-layer to add shadow on the original image
    image = Image.fromarray(im)
    back = Image.new('RGBA', image.size)
    back.paste(image)

    width, height = image.size
    # Creating a blank image to draw on
    poly = Image.new('RGBA', (height, width))
    pdraw = ImageDraw.Draw(poly)

    # Following code gets the random coordinates on image and we also
    # make sure that the shadow is within the interested portion of the image
    htlimit = height - np.random.random_integers(0, 20)
    wdlimit = width - np.random.random_integers(0, 20)

    x1 = np.random.random_integers(0, wdlimit)
    y1 = np.random.random_integers(50, htlimit)
    x2 = np.random.random_integers(50, wdlimit)
    y2 = np.random.random_integers(100, htlimit)

    x3 = np.random.random_integers(0, wdlimit)
    y3 = np.random.random_integers(0, htlimit)
    x4 = np.random.random_integers(100, wdlimit)
    y4 = np.random.random_integers(50, htlimit)

    # Draw polygon based on the random coordinates
    pdraw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                  fill=(100, 100, 100, 180), outline=(0, 0, 0, 0))

    # Pasting polygon on the original image
    back.paste(poly, (0, 0), mask=poly)
    npimage = np.asarray(back)
    rgb = cv2.cvtColor(npimage, cv2.COLOR_RGBA2RGB)
    return rgb

############################## Collecting all data from CSV #####################

lines = [] # Collection of data

# Collecting all recorded data into lines[]
with open ('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

with open ('more_driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Splitting training and validation samples
train_samples, validation_samples = train_test_split(lines, test_size=0.3)

############################## Defining generator #####################
# Following method defines the generator which can generate samples in small
# chunks and saves memory, I also process images to add 4 images in batch
# for each recorded data image

def generator(samples, batch_size=128):
    num_samples = len(samples)
    # batch_size is divided by 4, because we add 4 images per image
    batch_size = int(batch_size/4)

    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                # ALL_IMG folder contains all images collected for training
                name = 'ALL_IMG/'+batch_sample[0].split('/')[-1]
                # Fetch the original image
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                # Add the original image
                images.append(center_image)
                angles.append(center_angle)

                ##### Adding augmented images #########
                # Flip the image horizontally to make more data for car
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle * -1.0) # Negative steer, since image is horizontally flipped
                # Process image to add random brightness to simulate day-night visuals of track
                images.append(randomBrightnessImage(center_image))
                angles.append(center_angle)
                # Process image to add random polygon shadows on the track, to allow car
                # to easily differentiate between track and shadow
                images.append(shadowPolygon(center_image))
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            # Return samples, and yield
            yield sklearn.utils.shuffle(X_train, y_train)

############################## Model creation #####################

model=Sequential()
# Cropping image to process only the interested portion
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
# Normalizing the image
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# 5 Convolution layer with 'relu' for handling non linearity
model.add(Convolution2D(32, 5, 5))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))
model.add(MaxPooling2D((2,2)))
# Dropout avoids overfitting of training data and keeps model generalized
Dropout(0.2, noise_shape=None, seed=None)
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D((2,2)))
Dropout(0.3, noise_shape=None, seed=None)
model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

# Flatten the data and dense layers
model.add(Flatten())

model.add(Dense(80))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(10))

model.add(Dense(1))

############################## Model compilation and training #####################

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch= len(train_samples) * 4,
                    validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=8, verbose=1)

# Save the trained model
model.save('model.h5')
###################################################################################

import csv
import cv2
import numpy as np
#import sys

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dense
from scipy import ndimage
#import matplotlib.pyplot as plt

seed_value = 123
from numpy.random import seed
seed(seed_value)
from tensorflow import set_random_seed
set_random_seed(seed_value)

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    if line[0] == 'center':
        continue
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    #image = cv2.imread(current_path)
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

#NOTE: cv2.imread will get images in BGR format, while drive.py uses RGB. In
#the video above one way you could keep the same image formatting is to do
#"image = ndimage.imread(current_path)" with "from scipy import ndimage"
#instead.

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(-1.0*measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

def add_noise(input_image, mean=0, var=10):
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, input_image.shape)
    noisy_image = np.zeros(input_image.shape, np.float32)
    noisy_image[:, :, :] = input_image[:, :, :] + gaussian
    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    
    return noisy_image

def expand_dataset(X_input, y_input, threshold=0.25):
    X_duplicate = []
    y_duplicate = []
    for i in range(len(y_input)):
        i_input = i       
        y_input_val = abs(y_input[i])
        if y_input_val > threshold:
            augment_factor = int(16*abs(y_input_val))
        else:
            augment_factor = 0

        for _ in range(augment_factor):
            X_duplicate.append(add_noise(X_input[i_input, :, :, :]))
            y_duplicate.append(y_input[i_input])

    return X_duplicate, y_duplicate

X_duplicate, y_duplicate = expand_dataset(X_train, y_train, threshold=0.2)
print('Before nsamples=', len(y_train))
X_train = np.concatenate((X_train, X_duplicate), axis=0)
y_train = np.concatenate((y_train, y_duplicate), axis=0)
print('After nsamples=', len(y_train))

#fignum = 0
#fignum += 1
#plt.figure(fignum)
#plt.hist(y_train, bins=20)  # arguments are passed to np.histogram
#plt.title("Histogram for training data")
#plt.xlabel("Class label")
#plt.ylabel("Number of samples")
#plt.savefig("histogram_training.png")
#sys.exit(1)

model = Sequential()
model.add(Lambda(lambda x: (x - 128.0)/128.0, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, (5, 5), activation='relu', padding='valid'))
#model.add(Conv2D(24, (5, 5), activation='relu', padding='valid'))
model.add(Conv2D(36, (5, 5), activation='relu', padding='valid'))
model.add(Conv2D(48, (5, 5), activation='relu', padding='valid'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
#model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

#history_object = model.fit_generator(train_generator, samples_per_epoch =
#    len(train_samples), validation_data = 
#    validation_generator,
#    nb_val_samples = len(validation_samples), 
#    nb_epoch=5, verbose=1)

### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

model.save('model.h5')


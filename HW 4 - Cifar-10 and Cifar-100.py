import numpy as np
import tensorflow as tf
from tensorflow import keras

NUM_CLASSES = 10
lr = 1e-1

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train[10000:]
y_train = y_train[10000:]
x_valid = x_train[:10000]
y_valid = y_train[:10000]

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_valid = x_valid.reshape(x_valid.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_valid = tf.keras.utils.to_categorical(y_valid, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train)

model = keras.Sequential()
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=96,kernel_size=(3,3),input_shape=(32, 32, 3), strides=2,padding='valid',activation='relu', kernel_initializer=keras.initializers.VarianceScaling()))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Conv2D(filters=96,kernel_size=(3,3),strides=2,padding='valid',activation='relu', kernel_initializer=keras.initializers.VarianceScaling()))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Conv2D(filters=192,kernel_size=(3,3),strides=2,padding='valid',activation='relu', kernel_initializer=keras.initializers.VarianceScaling()))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Conv2D(filters=192,kernel_size=(3,3),strides=2,padding='valid',activation='relu', kernel_initializer=keras.initializers.VarianceScaling()))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer=keras.initializers.VarianceScaling()))

model.compile(optimizer=keras.optimizers.SGD(lr, momentum=0.7),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train, batch_size=512,epochs=50,validation_data=(x_valid, y_valid),verbose=2)

model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from google.colab import drive

drive.mount('/content/gdrive')

batch_size = 32

def addinception(grayimg):
    grayimg_inception = []
    for i in grayimg:
        #resize to 299,299,3 for input into inception
        i = resize(i, (299, 299, 3), mode='constant')
        grayimg_inception.append(i)
    grayimg_inception = np.array(grayimg_inception)
    #sets image to go from -1 to 1
    grayimg_inception = preprocess_input(grayimg_inception)
    with inception.graph.as_default():
        embed = inception.predict(grayimg_inception)
    return embed


def res(batch_size, x_train):
    for batch in datagen.flow(x_train, batch_size=batch_size):
        #sets grayscale across all three color channels
        grayscale = gray2rgb(rgb2gray(batch))
        embed = addinception(grayscale)
        labval = rgb2lab(batch)
        #grabs L value 
        L_val = labval[:,:,:,0]
        L_val = L_val.reshape(L_val.shape+(1,))
        #grabs ab value
        AB_val = labval[:,:,:,1:] / 128
        #returns format required for input and output of the neural net
        yield ([L_val, embed], AB_val)

inputimg = []
trainimg = 0
for filename in os.listdir('/content/gdrive/My Drive/images/Train/'):
    trainimg += 1
    inputimg.append(img_to_array(load_img('/content/gdrive/My Drive/images/Train/'+filename)))
    print("Training Image ", trainimg)
    # if (trainimg == 250):
    #     break

inputimg = np.array(inputimg, dtype=float)
#Normalize the image and make the values range from -1 to 1 (preprocessing)
x_train = inputimg / 255
x_train -= 0.5
x_train *=2

#Load inception network and imagenet weights
inceptionv2 = InceptionResNetV2(weights='imagenet', include_top=True)
inceptionv2.graph = tf.get_default_graph()

inceptionv2.layers.pop()
inceptionv2.outputs = [inceptionv2.layers[-1].output]
inceptionv2.layers[-1].outbound_nodes = []
inception = Sequential()
inception.add(inceptionv2)
inception.add(Dense(1001, activation='softmax'))

#Encoder Layer
encode = Input(shape=(256, 256, 1,))
encoder = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encode)
encoder = Conv2D(128, (3,3), activation='relu', padding='same')(encoder)
encoder = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder)
encoder = Conv2D(256, (3,3), activation='relu', padding='same')(encoder)
encoder = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder)
encoder = Conv2D(512, (3,3), activation='relu', padding='same')(encoder)
encoder = Conv2D(512, (3,3), activation='relu', padding='same')(encoder)
encoder = Conv2D(256, (3,3), activation='relu', padding='same')(encoder)

embedded_input = Input(shape=(1001,))

#Fusion Layer
fusion = RepeatVector(32 * 32)(embedded_input) 
fusion = Reshape(([32, 32, 1001]))(fusion)
fusion = concatenate([encoder, fusion], axis=3) 
fusion = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion) 

#Decoder Layer
decoder = Conv2D(128, (3,3), activation='relu', padding='same')(fusion)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2D(64, (3,3), activation='relu', padding='same')(decoder)
decoder = Conv2D(64, (3,3), activation='relu', padding='same')(decoder)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2D(32, (3,3), activation='relu', padding='same')(decoder)
decoder = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder)
decoder = UpSampling2D((2, 2))(decoder)

model = Model(inputs=[encode, embedded_input], outputs=decoder)
model = multi_gpu_model(model)

datagen = ImageDataGenerator(
        shear_range=0.4,
        zoom_range=0.4,
        rotation_range=40,
        horizontal_flip=True)

#Train model      
tensorboard = TensorBoard(log_dir="/content/gdrive/My Drive/images/output")
#model = load_model('images/weight/weights-improvement-200.hdf5')
model.compile(optimizer='adam', loss='mse')
filepath="/content/gdrive/My Drive/images/weight/weights-improvement-{epoch:02d}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model.summary()
model.fit_generator(res(batch_size, x_train), callbacks=[tensorboard, checkpoint], epochs=1000, steps_per_epoch=100)

test_images = []
testimg = 0
for filename in os.listdir('/content/gdrive/My Drive/images/Test/'):
    testimg += 1
    test_images.append(img_to_array(load_img('/content/gdrive/My Drive/images/Test/'+filename)))
    print("Test Image ", testimg)
    # if (testimg == 100):
    #     break

test_images = np.array(test_images, dtype=float)

#turns images into range from -1 to 1
test_images = 1.0/255*test_images
test_images -= 0.5
test_images *= 2

#turns all images into grayscale across all color channels so you can input grayscale or colored images to test with.
test_images = gray2rgb(rgb2gray(test_images))
test_images_embed = addinception(test_images)
test_images = rgb2lab(test_images)[:,:,:,0]
test_images = test_images.reshape(test_images.shape+(1,))

# Prediction
output = model.predict([test_images, test_images_embed])

#Restore AB values
output = output * 128

if not os.path.exists('result'):
    os.makedirs('result')

# Output
for i in range(len(output)):
    resimg = np.zeros((256, 256, 3))
    #Sets L channel
    resimg[:,:,0] = test_images[i][:,:,0]
    #Sets AB channel
    resimg[:,:,1:] = output[i]
    filename3 = os.path.join('/content/gdrive/My Drive/images/result/', 'img_'+str(i)+'.png')
    imsave(filename3, lab2rgb(resimg))

# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_20 (InputLayer)           (None, 256, 256, 1)  0                                            
# __________________________________________________________________________________________________
# conv2d_1555 (Conv2D)            (None, 128, 128, 64) 640         input_20[0][0]                   
# __________________________________________________________________________________________________
# conv2d_1556 (Conv2D)            (None, 128, 128, 128 73856       conv2d_1555[0][0]                
# __________________________________________________________________________________________________
# conv2d_1557 (Conv2D)            (None, 64, 64, 128)  147584      conv2d_1556[0][0]                
# __________________________________________________________________________________________________
# conv2d_1558 (Conv2D)            (None, 64, 64, 256)  295168      conv2d_1557[0][0]                
# __________________________________________________________________________________________________
# conv2d_1559 (Conv2D)            (None, 32, 32, 256)  590080      conv2d_1558[0][0]                
# __________________________________________________________________________________________________
# conv2d_1560 (Conv2D)            (None, 32, 32, 512)  1180160     conv2d_1559[0][0]                
# __________________________________________________________________________________________________
# input_21 (InputLayer)           (None, 1001)         0                                            
# __________________________________________________________________________________________________
# conv2d_1561 (Conv2D)            (None, 32, 32, 512)  2359808     conv2d_1560[0][0]                
# __________________________________________________________________________________________________
# repeat_vector_6 (RepeatVector)  (None, 1024, 1001)   0           input_21[0][0]                   
# __________________________________________________________________________________________________
# conv2d_1562 (Conv2D)            (None, 32, 32, 256)  1179904     conv2d_1561[0][0]                
# __________________________________________________________________________________________________
# reshape_6 (Reshape)             (None, 32, 32, 1001) 0           repeat_vector_6[0][0]            
# __________________________________________________________________________________________________
# concatenate_6 (Concatenate)     (None, 32, 32, 1257) 0           conv2d_1562[0][0]                
#                                                                  reshape_6[0][0]                  
# __________________________________________________________________________________________________
# conv2d_1563 (Conv2D)            (None, 32, 32, 256)  322048      concatenate_6[0][0]              
# __________________________________________________________________________________________________
# conv2d_1564 (Conv2D)            (None, 32, 32, 128)  295040      conv2d_1563[0][0]                
# __________________________________________________________________________________________________
# up_sampling2d_16 (UpSampling2D) (None, 64, 64, 128)  0           conv2d_1564[0][0]                
# __________________________________________________________________________________________________
# conv2d_1565 (Conv2D)            (None, 64, 64, 64)   73792       up_sampling2d_16[0][0]           
# __________________________________________________________________________________________________
# conv2d_1566 (Conv2D)            (None, 64, 64, 64)   36928       conv2d_1565[0][0]                
# __________________________________________________________________________________________________
# up_sampling2d_17 (UpSampling2D) (None, 128, 128, 64) 0           conv2d_1566[0][0]                
# __________________________________________________________________________________________________
# conv2d_1567 (Conv2D)            (None, 128, 128, 32) 18464       up_sampling2d_17[0][0]           
# __________________________________________________________________________________________________
# conv2d_1568 (Conv2D)            (None, 128, 128, 2)  578         conv2d_1567[0][0]                
# __________________________________________________________________________________________________
# up_sampling2d_18 (UpSampling2D) (None, 256, 256, 2)  0           conv2d_1568[0][0]                
# ==================================================================================================
# Total params: 6,574,050
# Trainable params: 6,574,050
# Non-trainable params: 0
# __________________________________________________________________________________________________
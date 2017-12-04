import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input,Activation
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.layers.normalization import BatchNormalization
K.set_image_dim_ordering('th')

np.random.seed(1000)

randomDim = 100

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, np.newaxis, :, :]

adam = Adam(lr=0.0002, beta_1=0.5)

# Generator
generator = Sequential()
generator.add(Dense(128*7*7, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.05)))
generator.add(Activation('relu'))
generator.add(Reshape((128, 7, 7)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator.add(Activation('relu'))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)

# Discriminator
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(1, 28, 28), kernel_initializer=initializers.RandomNormal(stddev=0.05)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.2))
discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.2))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

#D&G
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)

gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

d_loss = []
g_loss = []

def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(d_loss, label='D loss')
    plt.plot(g_loss, label='G loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('images/loss_vs_epoch_%d.png' % epoch)

def draw(epoch, examples=100, dim=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)

    plt.figure(figsize=(10, 10))
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/epoch_%d.png' % epoch)

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/g_epoch_%d.h5' % epoch)
    discriminator.save('models/d_epoch_%d.h5' % epoch)

def train(epochs=1, batchSize=128):
    batchCount = int(X_train.shape[0] / batchSize)
    print('Epochs:'+ str(epochs))
    print ('Batch size:' + str(batchSize))
    print ('Batches per epoch:'+ str(batchCount) )

    for epo in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % epo, '-'*15)
        for _ in tqdm(range(batchCount)):

            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dL = discriminator.fit(X, yDis, batch_size=128, epochs=1)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yD = np.ones(batchSize)
            discriminator.trainable = False
            gL = gan.fit(noise, yD, batch_size=128, epochs=1)

        d_loss.append(dL)
        g_loss.append(gL)
        print("discriminator loss: " + str(dL))
        print("generator loss: " + str(gL))
        if epo == 1 or epo % 5 == 0:
            draw(epo)
            saveModels(epo)

    plotLoss(epo)

if __name__ == '__main__':
    train(50, 128)

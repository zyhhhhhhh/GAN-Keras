import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import  Input, Activation
from keras.models import  Model, Sequential
from keras.layers.core import  Reshape, Dense
from keras.layers.core import  Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.layers.normalization import BatchNormalization
from tqdm import tqdm
K.set_image_dim_ordering('th')

np.random.seed(1000)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, np.newaxis, :, :]

adam = Adam(lr=0.0004, beta_1=0.5)

# G
G = Sequential()
G.add(Dense(128*7*7, input_dim=100, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
G.add(Activation('relu'))
G.add(Reshape((128, 7, 7)))
G.add(UpSampling2D(size=(2, 2)))
G.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
G.add(Activation('relu'))
G.add(UpSampling2D(size=(2, 2)))
G.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
G.compile(loss='binary_crossentropy', optimizer=adam)

# D
D = Sequential()
D.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(1, 28, 28), kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
D.add(LeakyReLU(0.2))
D.add(Dropout(0.2))
D.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
D.add(LeakyReLU(0.2))
D.add(Dropout(0.2))
D.add(Flatten())
D.add(Dense(1, activation='sigmoid'))
D.compile(loss='binary_crossentropy', optimizer=adam)

#D&G
D.trainable = False
In = Input(shape=(100,))
x = G(In)
Out = D(x)

gan = Model(inputs=In, outputs=Out)
gan.compile(loss='binary_crossentropy', optimizer=adam)

d_loss = []
g_loss = []

def drawFinal(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(d_loss, label='D loss')
    plt.plot(g_loss, label='G loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('images/loss_vs_epoch_%d.png' % epoch)

def draw(epoch, examples=100, dim=(10, 10)):
    randomIn = np.random.normal(0, 1, size=[examples, 100])
    out = G.predict(randomIn)
    plt.figure(figsize=(10, 10))
    for i in range(out.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(out[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/epoch_%d.png' % epoch)

# save models
def saveModels(epoch):
    G.save('models/g_epoch_%d.h5' % epoch)
    D.save('models/d_epoch_%d.h5' % epoch)

def trainModels(epochs):
    itr = int(X_train.shape[0] / 128)
    print('Epochs:'+ str(epochs))
    print ('Batch size:' + str(128))
    print ('Batches per epoch:'+ str(itr) )

    for epo in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % epo, '-'*15)
        for _ in tqdm(range(itr)):
            randomIn = np.random.normal(0, 1, size=[128, 100])
            imgIn = X_train[np.random.randint(0, X_train.shape[0], size=128)]

            out = G.predict(randomIn)
            X = np.concatenate([imgIn, out])
            yDis = np.zeros(2*128)
            yDis[:128] = 0.9

            D.trainable = True
            dL = D.fit(X, yDis, batch_size=128, epochs=1)

            # trainModels G
            randomIn = np.random.normal(0, 1, size=[128, 100])
            yD = np.ones(128)
            D.trainable = False
            gL = gan.fit(randomIn, yD, batch_size=128, epochs=1)

        d_loss.append(dL)
        g_loss.append(gL)
        print("D loss: " + str(dL))
        print("G loss: " + str(gL))
        if epo == 1 or epo % 5 == 0:
            draw(epo)
            saveModels(epo)

    drawFinal(epo)

if __name__ == '__main__':
    trainModels(50)

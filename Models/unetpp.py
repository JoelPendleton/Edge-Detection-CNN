import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import concatenate
import numpy as np
from PIL import Image
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, Lambda

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import matplotlib.pyplot as plt




def conv2d(filters: int):
    return Conv2D(filters=filters,
                  kernel_size=(3, 3),
                  padding='same',
                  kernel_regularizer=l2(0.),
                  bias_regularizer=l2(0.))


def conv2dtranspose(filters: int):
    return Conv2DTranspose(filters=filters,
                           kernel_size=(2, 2),
                           strides=(2, 2),
                           padding='same')


class UNetPP:
    """
    This is a class for edge-detection convolutional neural network using UNet++ architecture.

    Attributes:
        IMG_WIDTH (int): the width of the input images in pixels
        IMG_HEIGHT (int): the height of the input images in pixels
        IMG_CHANNELS (int): the number of colour channels of images
        number_of_filters (int): the number of filters to use in the model
        N_test (int): the number of examples in the test set
        N_train (int):  the number of examples in the training set
    """

    # Set some parameters
    number_of_filters = 16
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    IMG_CHANNELS = 1
    batch_size = 2

    N_test = len(os.listdir('./Data/Test/Input'))  # Number of test examples
    N_train = len(os.listdir('./Data/Train/Input'))  # Number of training examples

    def __init__(self):
        """
       The constructor for CNN class.

       Parameters:
          model (object): object containing all the information to utilise the neural network.
       """
        mirrored_strategy = tf.distribute.MirroredStrategy()
        model_exists = os.path.exists('./Checkpoints/model_unetpp_checkpoint.h5')

        if model_exists:  # If model has already been trained, load model
            with mirrored_strategy.scope():

                self.model = load_model('./Checkpoints/model_unetpp_checkpoint.h5')
        else:  # If model hasn't been trained create model
            with mirrored_strategy.scope():

                inputs = Input((self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS))
                s = Lambda(lambda x: x / 255)(inputs)

                x00 = conv2d(filters=int(16 * self.number_of_filters))(s)
                x00 = BatchNormalization()(x00)
                x00 = LeakyReLU(0.01)(x00)
                x00 = Dropout(0.2)(x00)
                x00 = conv2d(filters=int(16 * self.number_of_filters))(x00)
                x00 = BatchNormalization()(x00)
                x00 = LeakyReLU(0.01)(x00)
                x00 = Dropout(0.2)(x00)
                p0 = MaxPooling2D(pool_size=(2, 2))(x00)

                x10 = conv2d(filters=int(32 * self.number_of_filters))(p0)
                x10 = BatchNormalization()(x10)
                x10 = LeakyReLU(0.01)(x10)
                x10 = Dropout(0.2)(x10)
                x10 = conv2d(filters=int(32 * self.number_of_filters))(x10)
                x10 = BatchNormalization()(x10)
                x10 = LeakyReLU(0.01)(x10)
                x10 = Dropout(0.2)(x10)
                p1 = MaxPooling2D(pool_size=(2, 2))(x10)

                x01 = conv2dtranspose(int(16 * self.number_of_filters))(x10)
                x01 = concatenate([x00, x01])
                x01 = conv2d(filters=int(16 * self.number_of_filters))(x01)
                x01 = BatchNormalization()(x01)
                x01 = LeakyReLU(0.01)(x01)
                x01 = conv2d(filters=int(16 * self.number_of_filters))(x01)
                x01 = BatchNormalization()(x01)
                x01 = LeakyReLU(0.01)(x01)
                x01 = Dropout(0.2)(x01)

                x20 = conv2d(filters=int(64 * self.number_of_filters))(p1)
                x20 = BatchNormalization()(x20)
                x20 = LeakyReLU(0.01)(x20)
                x20 = Dropout(0.2)(x20)
                x20 = conv2d(filters=int(64 * self.number_of_filters))(x20)
                x20 = BatchNormalization()(x20)
                x20 = LeakyReLU(0.01)(x20)
                x20 = Dropout(0.2)(x20)
                p2 = MaxPooling2D(pool_size=(2, 2))(x20)

                x11 = conv2dtranspose(int(16 * self.number_of_filters))(x20)
                x11 = concatenate([x10, x11])
                x11 = conv2d(filters=int(16 * self.number_of_filters))(x11)
                x11 = BatchNormalization()(x11)
                x11 = LeakyReLU(0.01)(x11)
                x11 = conv2d(filters=int(16 * self.number_of_filters))(x11)
                x11 = BatchNormalization()(x11)
                x11 = LeakyReLU(0.01)(x11)
                x11 = Dropout(0.2)(x11)

                x02 = conv2dtranspose(int(16 * self.number_of_filters))(x11)
                x02 = concatenate([x00, x01, x02])
                x02 = conv2d(filters=int(16 * self.number_of_filters))(x02)
                x02 = BatchNormalization()(x02)
                x02 = LeakyReLU(0.01)(x02)
                x02 = conv2d(filters=int(16 * self.number_of_filters))(x02)
                x02 = BatchNormalization()(x02)
                x02 = LeakyReLU(0.01)(x02)
                x02 = Dropout(0.2)(x02)

                x30 = conv2d(filters=int(128 * self.number_of_filters))(p2)
                x30 = BatchNormalization()(x30)
                x30 = LeakyReLU(0.01)(x30)
                x30 = Dropout(0.2)(x30)
                x30 = conv2d(filters=int(128 * self.number_of_filters))(x30)
                x30 = BatchNormalization()(x30)
                x30 = LeakyReLU(0.01)(x30)
                x30 = Dropout(0.2)(x30)
                p3 = MaxPooling2D(pool_size=(2, 2))(x30)

                x21 = conv2dtranspose(int(16 * self.number_of_filters))(x30)
                x21 = concatenate([x20, x21])
                x21 = conv2d(filters=int(16 * self.number_of_filters))(x21)
                x21 = BatchNormalization()(x21)
                x21 = LeakyReLU(0.01)(x21)
                x21 = conv2d(filters=int(16 * self.number_of_filters))(x21)
                x21 = BatchNormalization()(x21)
                x21 = LeakyReLU(0.01)(x21)
                x21 = Dropout(0.2)(x21)

                x12 = conv2dtranspose(int(16 * self.number_of_filters))(x21)
                x12 = concatenate([x10, x11, x12])
                x12 = conv2d(filters=int(16 * self.number_of_filters))(x12)
                x12 = BatchNormalization()(x12)
                x12 = LeakyReLU(0.01)(x12)
                x12 = conv2d(filters=int(16 * self.number_of_filters))(x12)
                x12 = BatchNormalization()(x12)
                x12 = LeakyReLU(0.01)(x12)
                x12 = Dropout(0.2)(x12)

                x03 = conv2dtranspose(int(16 * self.number_of_filters))(x12)
                x03 = concatenate([x00, x01, x02, x03])
                x03 = conv2d(filters=int(16 * self.number_of_filters))(x03)
                x03 = BatchNormalization()(x03)
                x03 = LeakyReLU(0.01)(x03)
                x03 = conv2d(filters=int(16 * self.number_of_filters))(x03)
                x03 = BatchNormalization()(x03)
                x03 = LeakyReLU(0.01)(x03)
                x03 = Dropout(0.2)(x03)

                m = conv2d(filters=int(256 * self.number_of_filters))(p3)
                m = BatchNormalization()(m)
                m = LeakyReLU(0.01)(m)
                m = conv2d(filters=int(256 * self.number_of_filters))(m)
                m = BatchNormalization()(m)
                m = LeakyReLU(0.01)(m)
                m = Dropout(0.2)(m)

                x31 = conv2dtranspose(int(128 * self.number_of_filters))(m)
                x31 = concatenate([x31, x30])
                x31 = conv2d(filters=int(128 * self.number_of_filters))(x31)
                x31 = BatchNormalization()(x31)
                x31 = LeakyReLU(0.01)(x31)
                x31 = conv2d(filters=int(128 * self.number_of_filters))(x31)
                x31 = BatchNormalization()(x31)
                x31 = LeakyReLU(0.01)(x31)
                x31 = Dropout(0.2)(x31)

                x22 = conv2dtranspose(int(64 * self.number_of_filters))(x31)
                x22 = concatenate([x22, x20, x21])
                x22 = conv2d(filters=int(64 * self.number_of_filters))(x22)
                x22 = BatchNormalization()(x22)
                x22 = LeakyReLU(0.01)(x22)
                x22 = conv2d(filters=int(64 * self.number_of_filters))(x22)
                x22 = BatchNormalization()(x22)
                x22 = LeakyReLU(0.01)(x22)
                x22 = Dropout(0.2)(x22)

                x13 = conv2dtranspose(int(32 * self.number_of_filters))(x22)
                x13 = concatenate([x13, x10, x11, x12])
                x13 = conv2d(filters=int(32 * self.number_of_filters))(x13)
                x13 = BatchNormalization()(x13)
                x13 = LeakyReLU(0.01)(x13)
                x13 = conv2d(filters=int(32 * self.number_of_filters))(x13)
                x13 = BatchNormalization()(x13)
                x13 = LeakyReLU(0.01)(x13)
                x13 = Dropout(0.2)(x13)

                x04 = conv2dtranspose(int(16 * self.number_of_filters))(x13)
                x04 = concatenate([x04, x00, x01, x02, x03], axis=3)
                x04 = conv2d(filters=int(16 * self.number_of_filters))(x04)
                x04 = BatchNormalization()(x04)
                x04 = LeakyReLU(0.01)(x04)
                x04 = conv2d(filters=int(16 * self.number_of_filters))(x04)
                x04 = BatchNormalization()(x04)
                x04 = LeakyReLU(0.01)(x04)
                x04 = Dropout(0.2)(x04)

                outputs = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(x04)

                self.model = Model(inputs=[inputs], outputs=[outputs])

            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



    def load_training_set(self):
        """
        The function to load training examples for CNN and put them inside self.X_train and self.Y_train.

        Returns:
           True upon completion
        """
        # Define dimensions of examples
        self.X_train = np.zeros((self.N_train, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        self.Y_train = np.zeros((self.N_train, self.IMG_HEIGHT, self.IMG_WIDTH,  1))

        # Load in training examples
        for i in range(self.N_train):
            x_image = Image.open('./Data/Train/Input/input_{0}.png'.format(i + 1)).convert("L").resize(
                (self.IMG_WIDTH, self.IMG_HEIGHT))
            x = np.array(x_image)
            self.X_train[i] = x

            y_image = Image.open('./Data/Train/Output/output_{0}.png'.format(i + 1)).convert("L").resize(
                (self.IMG_WIDTH, self.IMG_HEIGHT))
            # convert("L") reduces to single channel greyscale, resize reduces resolution to IMG_WIDTH x IMG_HEIGHT
            y = (np.array(y_image) / 255 == 1)  # divide by 255 as np.array puts white as 255 and black as 0.
            # Use == 1 to convert to boolean
            y = np.reshape(np.array(y), (self.IMG_HEIGHT, self.IMG_WIDTH,  1))
            self.Y_train[i] = y  # Add training output to array

        return True

    def load_test_set(self):
        """
        The function to load the test set for CNN, and place it inside self.X_test and self.Y_test.

        Returns:
           True upon completion
        """
        self.X_test = np.zeros((self.N_train, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        self.Y_test = np.zeros((self.N_train, self.IMG_HEIGHT, self.IMG_WIDTH, 1))

        # Load in test set

        for i in range(self.N_test):
            x_image = Image.open('./Data/Test/Input/input_{0}.png'.format(i + 1)).convert("L").resize(
                (self.IMG_WIDTH, self.IMG_HEIGHT))
            x = np.array(x_image)
            self.X_test[i] = x

            y_image = Image.open('./Data/Test/Output/output_{0}.png'.format(i + 1)).convert("L").resize(
                (self.IMG_WIDTH, self.IMG_HEIGHT))
            # convert("L") reduces to single channel greyscale, resize reduces resolution to IMG_WIDTH x IMG_HEIGHT
            y = (np.array(y_image) / 255 == 1)  # divide by 255 as np.array puts white as 255 and black as 0.
            # Use == 1 to convert to boolean
            y = np.reshape(np.array(y), (self.IMG_HEIGHT, self.IMG_WIDTH, 1))
            self.Y_test[i] = y  # Add training output to array

        return True

    def train(self):
        """
          The function to train the CNN using training examples.

          Returns:
              results: the results of the trained CNN.
          """
        self.load_training_set()
        earlystopper = EarlyStopping(patience=10, verbose=1)
        checkpointer = ModelCheckpoint('./Checkpoints/model_unetpp_checkpoint.h5', verbose=1, save_best_only=True)
        results = self.model.fit(self.X_train, self.Y_train,
                              validation_split=0.05,callbacks=[earlystopper, checkpointer],
                              batch_size=self.batch_size, use_multiprocessing=True,
                              epochs=100,
                              shuffle=True)

        print("Program finished running. The CNN has been trained.")

        return results


    def predict(self):
        """
        The function to make predictions on training examples.

        Returns:
            [preds_train_t, preds_val_t, preds_test_t] (boolean): The boolean True or False predictions of positions
            of black and white pixels edge or no edge, for each of the data sets.
        """

        # if not os.path.exists("./Data/Train/Prediction"):
        #     os.makedirs("./Data/Train/Prediction")
        #
        # if not os.path.exists("./Data/Test/Prediction"):
        #     os.makedirs("./Data/Test/Prediction")
        #
        # self.load_training_set()
        # self.load_test_set()
        #
        # # Predict on train, val and test
        # preds_train = self.model.predict(self.X_train[:int(self.X_train.shape[0] * 0.9)], verbose=1)
        # preds_val = self.model.predict(self.X_train[int(self.X_train.shape[0] * 0.9):], verbose=1)
        # preds_test = self.model.predict(self.X_test, verbose=1)
        #
        # # Threshold predictions
        # preds_train_t = (preds_train > 0.5).astype(np.uint8)
        # preds_val_t = (preds_val > 0.5).astype(np.uint8)
        # preds_test_t = (preds_test > 0.5).astype(np.uint8)
        #
        # # Save training set predictions
        # for i in range(len(preds_train)):
        #     plt.imsave("./Data/Train/Prediction/prediction_{0}.png".format(i+1), np.squeeze(preds_train_t[i]), cmap='gray')
        #
        # # Save val set predictions
        # for i in range(len(preds_val)):
        #     plt.imsave("./Data/Train/Prediction/prediction_{0}.png".format(i + len(preds_train)),
        #                np.squeeze(preds_val_t[i]), cmap='gray')
        #
        # # Save test set predictions
        # for i in range(len(preds_test)):
        #     plt.imsave("./Data/Test/Prediction/prediction_{0}.png".format(i + 1),
        #                np.squeeze(preds_test_t[i]), cmap='gray')
        #
        # print("Program finished running. Predictions saved.")
        #
        # return preds_train_t, preds_val_t, preds_test_t



        # Load real data

        N_real_data = len(os.listdir('./Data/Real_Data/Input'))  # Number of test examples
        X_real = np.zeros((N_real_data, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))

        for i in range(N_real_data):
            x_image = Image.open('./Data/Real_Data/Input/real_data_{0}.png'.format(i + 1)).convert("RGB").resize(
                (self.IMG_WIDTH, self.IMG_HEIGHT))
            x = np.array(x_image)
            X_real[i] = x

        preds_real = self.model.predict(X_real, verbose=1)
        preds_real_t = (preds_real > 0.5).astype(np.uint8)

        # Save real data predictions
        for i in range(len(preds_real)):
            plt.imsave('./Data/Real_Data/Output/real_data_{0}.png'.format(i+1), np.squeeze(preds_real_t[i]), cmap='gray')

        return True

    def evaluate(self):
        """
        The function to make evaluate model on test set.

        Returns:
            self.model.evaluate: The evaluated metrics of the model's performance using the test set.
        """
        self.load_test_set()

        return self.model.evaluate(self.X_test[:1000], self.Y_test[:1000], use_multiprocessing = True, batch_size=self.batch_size)

    def summary(self):
        """
        The function to output summary of model.

        Returns:
            self.model.summary(): summary of model
        """
        return self.model.summary()





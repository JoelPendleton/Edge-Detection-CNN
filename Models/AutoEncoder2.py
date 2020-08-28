import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Lambda, UpSampling2D
import tensorflow as tf
from datetime import datetime
from PIL import Image
import os.path
import random


class AutoEncoder2:
    """
    This is a class for edge-detection convolutional neural network using original AutoEncoder1 architecture.

    Attributes:
        IMG_WIDTH (int): the width of the input images in pixels
        IMG_HEIGHT (int): the height of the input images in pixels
        IMG_CHANNELS (int): the number of colour channels of images

    """

    # Set some parameters
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    IMG_CHANNELS = 1

    N_test = len(os.listdir('./Data/Test/Input'))  # Number of test examples
    N_train = len(os.listdir('./Data/Train/Input'))  # Number of training examples

    def __init__(self):
        """
       The constructor for CNN class.

       Parameters:
          model (object): object containing all the information to utilise the neural network.
       """
        # seed random number generator
        random.seed(datetime.now())  # use current time as random number seed

        model_exists = os.path.exists('./Checkpoints/model_autoencoder2_checkpoint.h5')
        mirrored_strategy = tf.distribute.MirroredStrategy()

        if model_exists:  # If model has already been trained, load model
            with mirrored_strategy.scope():
                self.model = load_model('./Checkpoints/model_autoencoder2_checkpoint.h5')
        else:  # If model hasn't been trained create model
            with mirrored_strategy.scope():

                # Build AutoEncoder2 model
                inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
                s = Lambda(lambda x: x / 255)(inputs)

                x = Conv2D(32, (3, 3), activation='relu', padding ='same')(s)
                x = MaxPool2D((2, 2), padding='same')(x)
                x = Conv2D(32, (3, 3), activation='relu', padding ='same')(x)
                encoded = MaxPool2D((2, 2), padding='same')(x)
                x = Conv2D(32, (3, 3), activation='relu', padding ='same')(encoded)
                x = UpSampling2D((2, 2))(x)
                x = Conv2D(32, (3, 3), activation='relu', padding ='same')(x)
                x = UpSampling2D((2, 2))(x)
                outputs = Conv2D(1, (3, 3), activation='sigmoid', padding ='same')(x)

                self.model = Model(inputs=[inputs], outputs=[outputs])

            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    def load_examples(self):
        """
        The function to load training examples for CNN.

        Returns:
            self.X_train.shape (int): the shape of the training example array.
        """
        # Define dimensions of examples
        self.X_test = np.zeros((self.N_train, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
        self.X_train = np.zeros((self.N_train, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
        self.Y_train = np.zeros((self.N_train, self.IMG_HEIGHT, self.IMG_WIDTH, 1), dtype=bool)

        # Load in training examples

        for i in range(self.N_train):

            x_image = Image.open('../Data/Train/Input/input_{0}.png'.format(i+1)).convert("L").resize((self.IMG_WIDTH, self.IMG_HEIGHT))
            x = np.reshape(np.array(x_image), (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS) )#np.array(x_image)
            self.X_train[i] = x

            y_image = Image.open('../Data/Train/Output/output_{0}.png'.format(i+1)).convert("L").resize((self.IMG_WIDTH, self.IMG_HEIGHT))
            # convert("L") reduces to single channel greyscale, resize reduces resolution to IMG_WIDTH x IMG_HEIGHT
            y = (np.array(y_image) / 255 == 1)  # divide by 255 as np.array puts white as 255 and black as 0.
            # Use == 1 to convert to boolean
            self.Y_train[i] = y[:, :, np.newaxis]  # Add training output to array

        return self.X_train.shape


    def train(self):
        """
        The function to train the CNN using training examples.

        Returns:
            results (object): the results of the trained CNN.
        """
        earlystopper = EarlyStopping(patience=15, verbose=1)
        checkpointer = ModelCheckpoint('./Checkpoints/model_autoencoder2_checkpoint.h5', verbose=1, save_best_only=True)
        results = self.model.fit(self.X_train, self.Y_train, validation_split=0.1, batch_size=64, epochs=100,
                                 shuffle=True, use_multiprocessing=True, callbacks=[earlystopper, checkpointer])

        print("Program finished running. The CNN has been trained.")

        return results

    def predict(self):
        """
        The function to make predictions on training examples.

        Returns:
            preds_train_t (float): The binary True or False predictions of positions of black and white pixels /
            edge or no edge.
        """
        if not os.path.exists("../Data/Train/Prediction"):
            os.makedirs("../Data/Train/Prediction")

        if not os.path.exists("../Data/Test/Prediction"):
            os.makedirs("../Data/Test/Prediction")



        # Predict on train, val and test
        preds_train = self.model.predict(self.X_train[:int(self.X_train.shape[0] * 0.9)], verbose=1)
        preds_val = self.model.predict(self.X_train[int(self.X_train.shape[0] * 0.9):], verbose=1)
        preds_test = self.model.predict(self.X_test, verbose=1)


        # Threshold predictions
        preds_train_t = (preds_train > 0.5).astype(np.uint8)
        preds_val_t = (preds_val > 0.5).astype(np.uint8)
        preds_test_t = (preds_test > 0.5).astype(np.uint8)

        # Save training set predictions
        for i in range(len(preds_train)):
            plt.imsave("../Data/Train/Prediction/prediction_{0}.png".format(i+1), np.squeeze(preds_train_t[i]), cmap='gray')

        # Save val set predictions
        for i in range(len(preds_val)):
            plt.imsave("../Data/Train/Prediction/prediction_{0}.png".format(i + len(preds_train)),
                       np.squeeze(preds_val_t[i]), cmap='gray')

        # Save test set predictions
        for i in range(len(preds_test)):
            plt.imsave("../Data/Test/Prediction/prediction_{0}.png".format(i + 1),
                       np.squeeze(preds_test_t[i]), cmap='gray')

        print("Program finished running. Predictions saved.")

        return preds_train_t

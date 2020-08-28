import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Flatten, Reshape, Conv2DTranspose, LeakyReLU
from datetime import datetime
from PIL import Image
import os.path
import random


class AutoEncoder1:
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

    N_test = len(os.listdir('../Data/Test/Input'))  # Number of test examples
    N_train = len(os.listdir('../Data/Train/Input'))  # Number of training examples


    def __init__(self):
        """
       The constructor for CNN class.

       Parameters:
          model (object): object containing all the information to utilise the neural network.
       """
        # seed random number generator
        random.seed(datetime.now())  # use current time as random number seed

        model_exists = os.path.exists('../model_autoencoder1_checkpoint.h5')

        if model_exists:  # If model has already been trained, load model
            self.model = load_model('../model_autoencoder1_checkpoint.h5')
        else:  # If model hasn't been trained create model
            latent_dim = 128
            # Build U-Net model
            inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
            s = Lambda(lambda x: x / 255)(inputs)


            e = Conv2D(32, (3, 3), padding="same")(s)
            e = BatchNormalization()(e)
            e = LeakyReLU(alpha=0.2)(e)
            e = MaxPool2D((2, 2))(e)
            e = Conv2D(64, (3, 3), padding="same")(e)
            e = BatchNormalization()(e)
            e = LeakyReLU(alpha=0.2)(e)
            e = MaxPool2D((2, 2))(e)
            l = Flatten()(e)
            units = e.shape[1]
            l = Dense(latent_dim, name="latent")(l)
            l = Dense(units)(l)
            l = LeakyReLU(alpha=0.2)(l)

            d = Reshape((128, 128, 64))(l)
            d = Conv2DTranspose(64, (3, 3), strides=2, padding="same")(d)
            d = BatchNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            d = Conv2DTranspose(1, (3, 3), strides=2, padding="same")(d)
            d = BatchNormalization()(d)
            outputs = Activation("sigmoid", name="outputs")(d)


            self.model = Model(inputs=[inputs], outputs=[outputs])
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    def load_examples(self):
        """
        The function to load training examples for CNN.

        Returns:
            self.X_train.shape (int): the shape of the training example array.
        """
        # Define dimensions of training examples
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
        checkpointer = ModelCheckpoint('../model_autoencoder1_checkpoint.h5', verbose=1, save_best_only=True)
        results = self.model.fit(self.X_train, self.Y_train, validation_split=0.1, batch_size=16, epochs=100,
                            callbacks=[earlystopper, checkpointer])

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

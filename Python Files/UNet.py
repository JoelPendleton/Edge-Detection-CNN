import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from PIL import Image
import os.path
import random


class UNet:
    """
    This is a class for edge-detection convolutional neural network using UNet architecture.

    Attributes:
        IMG_WIDTH (int): the width of the input images in pixels
        IMG_HEIGHT (int): the height of the input images in pixels
        IMG_CHANNELS (int): the number of colour channels of images

    """

    path = '../Data/Train/Input'
    num_files = len(os.listdir(path))

    # Set some parameters
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    IMG_CHANNELS = 3

    N_train = num_files  # Number of training examples

    def __init__(self):
        """
       The constructor for CNN class.

       Parameters:
          model (object): object containing all the information to utilise the neural network.
       """
        # seed random number generator
        random.seed(datetime.now())  # use current time as random number seed

        model_exists = os.path.exists('../model_unet_checkpoint.h5')

        if model_exists:  # If model has already been trained, load model
            self.model = load_model('../model_unet_checkpoint.h5')
        else:  # If model hasn't been trained create model

            # Build U-Net model
            inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
            s = Lambda(lambda x: x / 255)(inputs)

            c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
            c1 = BatchNormalization()(c1)
            c1 = Dropout(0.1)(c1)
            c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
            c1 = BatchNormalization()(c1)
            p1 = MaxPooling2D((2, 2))(c1)

            c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
            c2 = BatchNormalization()(c2)
            c2 = Dropout(0.1)(c2)
            c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
            c2 = BatchNormalization()(c2)
            p2 = MaxPooling2D((2, 2))(c2)

            c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
            c3 = BatchNormalization()(c3)
            c3 = Dropout(0.2)(c3)
            c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
            c3 = BatchNormalization()(c3)
            p3 = MaxPooling2D((2, 2))(c3)

            c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
            c4 = BatchNormalization()(c4)
            c4 = Dropout(0.2)(c4)
            c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
            c4 = BatchNormalization()(c4)
            p4 = MaxPooling2D(pool_size=(2, 2))(c4)

            c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
            c5 = BatchNormalization()(c5)
            c5 = Dropout(0.3)(c5)
            c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
            c5 = BatchNormalization()(c5)

            u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
            u6 = concatenate([u6, c4])
            c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
            c6 = BatchNormalization()(c6)
            c6 = Dropout(0.2)(c6)
            c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
            c6 = BatchNormalization()(c6)

            u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
            u7 = concatenate([u7, c3])
            c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
            c7 = BatchNormalization()(c7)
            c7 = Dropout(0.2)(c7)
            c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
            c7 = BatchNormalization()(c7)

            u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
            u8 = concatenate([u8, c2])
            c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
            c8 = BatchNormalization()(c8)
            c8 = Dropout(0.1)(c8)
            c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
            c8 = BatchNormalization()(c8)

            u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
            u9 = concatenate([u9, c1], axis=3)
            c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
            c9 = BatchNormalization()(c9)
            c9 = Dropout(0.1)(c9)
            c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
            c9 = BatchNormalization()(c9)

            outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
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

            x_image = Image.open('../Data/Train/Input/input_{0}.png'.format(i+1)).convert("RGB").resize((self.IMG_WIDTH, self.IMG_HEIGHT))
            x = np.array(x_image)
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
        checkpointer = ModelCheckpoint('../model_unet_checkpoint.h5', verbose=1, save_best_only=True)
        results = self.model.fit(self.X_train, self.Y_train, validation_split=0.1, batch_size=20, epochs=100,
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
        #preds_test = self.model.predict(self.X_test, verbose=1)

        # Threshold predictions
        preds_train_t = (preds_train > 0.5).astype(np.uint8)
        preds_val_t = (preds_val > 0.5).astype(np.uint8)
        #preds_test_t = (preds_test > 0.5).astype(np.uint8)

        # Save training set predictions
        for i in range(len(preds_train)):
            plt.imsave("../Data/Train/Prediction/prediction_{0}.png".format(i+1), np.squeeze(preds_train_t[i]), cmap='gray')

        # Save val set predictions
        for i in range(len(preds_train)-1, len(preds_train) + len(preds_val)):
            plt.imsave("../Data/Train/Prediction/prediction_{0}.png".format(i + 1),
                       np.squeeze(preds_val_t[i]), cmap='gray')

        # # Save test set predictions
        # for i in range(len(preds_test)):
        #     plt.imsave("../Test/Prediction/prediction_{0}.png".format(i + 1),
        #                np.squeeze(preds_test_t[i]), cmap='gray')

        print("Program finished running. Training example predictions saved in Training_Prediction folder.")

        return preds_train_t

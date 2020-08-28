
from datetime import datetime
import random
import sys
from Models.UNet import UNet
from Models.AutoEncoder1 import AutoEncoder1
from Models.AutoEncoder2 import AutoEncoder2
from Models.UNetPP.unetpp import UNetPP

try:

    command = sys.argv[1]
    if command == '--help':
        print("Make sure when you call this file you pass one of the following flags.\n"
              "--train trains CNN on examples\n"
              "--predict generates predictions on training examples and validation set using CNN\n"
              "--summary summaries convolutional neural network architecture")
    elif command == "--train" or command == "--predict":
        architecture = sys.argv[2]

        # seed random number generator
        random.seed(datetime.now())  # use current time as random number seed

        if architecture == '--unet':
            UNet = UNet()
            UNet.load_examples()
        elif architecture == '--autoencoder1':
            AutoEncoder1.load_examples()
            AutoEncoder1 = AutoEncoder1()

        elif architecture == '--autoencoder2':
            AutoEncoder2 = AutoEncoder2()
            AutoEncoder2.load_examples()
        elif architecture == '--UNet++':
            UNetPP = UNetPP()
            UNetPP.load_examples()


        if command == "--train":
            if architecture == '--unet':
                UNet.train()
            elif architecture == '--autoencoder1':
                AutoEncoder1.train()
            elif architecture == '--autoencoder2':
                AutoEncoder2.train()
            elif architecture == '--UNet++':
                UNetPP.train()



        if command == "--predict":
            if architecture == '--unet':
                UNet.predict()
            elif architecture == '--autoencoder1':
                AutoEncoder1.predict()
            elif architecture == '--autoencoder2':
                AutoEncoder2.predict()
            elif architecture == '--UNet++':
                UNetPP.predict()


    elif command == "--summary":
        architecture = sys.argv[2]

        if architecture == '--unet':
            UNet = UNet()
            UNet.model.summary()
        elif architecture == '--autoencoder1':
            AutoEncoder1 = AutoEncoder1()
            AutoEncoder1.model.summary()
        elif architecture == '--autoencoder2':
            AutoEncoder2 = AutoEncoder2()
            AutoEncoder2.model.summary()
        elif architecture == '--UNet++':
            UNetPP = UNetPP()
            UNetPP.model.summary()




    else:
        print("Make sure when you call this file you pass one of the following flags.\n"
              "--train trains CNN on examples\n"
              "--predict generates predictions on training examples and validation set using CNN\n"
              "--summary summaries convolutional neural network architecture")
except IndexError:

    print("An Index error has occured!")
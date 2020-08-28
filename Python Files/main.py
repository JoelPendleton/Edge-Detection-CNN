
from datetime import datetime
import random
import sys
from UNet import UNet
from AutoEncoder1 import AutoEncoder1
from AutoEncoder2 import AutoEncoder2



try:
    command = sys.argv[1]
    if command == '--help':
        print("Make sure when you call this file you pass one of the following flags.\n"
              "--train trains CNN on examples\n"
              "--predict generates predictions on training examples and validation set using CNN\n"
              "--summary summaries convolutional neural network architecture")
    elif command == "--train" or command == "--predict":
        architecture = sys.argv[2]

        if architecture == '--unet':
            UNet = UNet()
        elif architecture == 'autoencoder1':
            AutoEncoder1 = AutoEncoder1()
        elif architecture == 'autoencoder2':
            AutoEncoder2 = AutoEncoder2()


        # seed random number generator
        random.seed(datetime.now())  # use current time as random number seed
        #UNet.load_examples()
        #AutoEncoder1.load_examples()
        AutoEncoder2.load_examples()


        if command == "--train":
            if architecture == '--unet':
                UNet.train()
            elif architecture == 'autoencoder1':
                AutoEncoder1.train()
            elif architecture == 'autoencoder2':
                AutoEncoder2.train()



        if command == "--predict":
            if architecture == '--unet':
                UNet.predict()
            elif architecture == 'autoencoder1':
                AutoEncoder1.predict()
            elif architecture == 'autoencoder2':
                AutoEncoder2.predict()


    elif command == "--summary":
        architecture = sys.argv[2]

        if architecture == '--unet':
            UNet = UNet()
            UNet.model.summary()
        elif architecture == 'autoencoder1':
            AutoEncoder1 = AutoEncoder1()
            AutoEncoder1.model.summary()
        elif architecture == 'autoencoder2':
            AutoEncoder2 = AutoEncoder2()
            AutoEncoder2.model.summary()




    else:
        print("Make sure when you call this file you pass one of the following flags.\n"
              "--train trains CNN on examples\n"
              "--predict generates predictions on training examples and validation set using CNN\n"
              "--summary summaries convolutional neural network architecture")
except IndexError:

    print("An Index error has occured!")
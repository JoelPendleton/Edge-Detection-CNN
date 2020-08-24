
from datetime import datetime
import random
import sys
from UNet import UNet

try:
    argument = sys.argv[1]
    if argument == '--help':
        print("Make sure when you call this file you pass one of the following flags.\n"
              "--train trains CNN on examples\n"
              "--predict generates predictions on training examples and validation set using CNN\n"
              "--summary summaries convolutional neural network architecture")
    elif argument == "--train" or argument == "--predict":
        UNet = UNet()
        # seed random number generator
        random.seed(datetime.now())  # use current time as random number seed
        UNet.load_examples()


        if argument == "--train":
            UNet.train()

        if argument == "--predict":
            UNet.predict()
    elif argument == "--summary":
        SegNet = UNet()
        SegNet.model.summary()


    else:
        print("Make sure when you call this file you pass one of the following flags.\n"
              "--train trains CNN on examples\n"
              "--predict generates predictions on training examples and validation set using CNN\n"
              "--summary summaries convolutional neural network architecture")
except IndexError:

    print("To run this program you must pass a flag:\n"
          "--train trains CNN on examples\n"
          "--predict generates predictions on training examples and validation set using CNN\n"
          "--summary summaries convolutional neural network architecture")
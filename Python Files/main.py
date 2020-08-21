
from datetime import datetime
import random
import sys
from CNN import CNN


try:
    argument = sys.argv[1]
    if argument == '--help':
        print("Make sure when you call this file you pass one of the following flags.\n"
              "--train trains CNN on examples\n"
              "--predict generates predictions on training examples and validation set using CNN\n"
              "--summary summaries convolutional neural network architecture")
    elif argument == "--train" or argument == "--predict":
        CNN = CNN()
        # seed random number generator
        random.seed(datetime.now())  # use current time as random number seed
        CNN.load_examples()


        if argument == "--train":
            CNN.train()

        if argument == "--predict":
            CNN.predict()
    elif argument == "--summary":
        CNN = CNN()
        CNN.model.summary()


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
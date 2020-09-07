import sys
from Models.UNet import UNet
from Models.UNetPP import UNetPP

try:
    command = sys.argv[1]
except IndexError:
    print("An Index error has occured! Check the documentation to make sure your passing all the required arguments.")

if command == '--Help':
    print("Check the documentation.")

elif command == "--Train" or command == "--Predict":
    try:
        architecture = sys.argv[2]
    except IndexError:
        print("An Index error has occured! Check the documentation to make sure your passing all the required arguments.")

    if command == "--Train":
        if architecture == '--UNet':
            UNet = UNet()
            UNet.train()
        elif architecture == '--UNet++':
            UNetPP = UNetPP()
            UNetPP.train()
        else:
            raise Exception( "You have passed an invalid argument.\nCheck the documentation for the allowed arguments.")

    elif command == "--Predict":
        if architecture == '--UNet':
            UNet = UNet()
            UNet.predict()
        elif architecture == '--UNet++':
            UNetPP = UNetPP()
            UNetPP.predict()
        else:
            raise Exception("You have passed an invalid argument.\nCheck the documentation for the allowed arguments.")
    else:
        raise Exception("You have passed an invalid argument.\nCheck the documentation for the allowed arguments.")


elif command == "--Summary":
    architecture = sys.argv[2]

    if architecture == '--UNet':
        UNet = UNet()
        UNet.summary()
    elif architecture == '--UNet++':
        UNetPP = UNetPP()
        UNetPP.summary()
    else:
        raise Exception("You have passed an invalid argument.\nCheck the documentation for the allowed arguments.")

elif command == "--Evaluate":
    architecture = sys.argv[2]

    if architecture == '--UNet':
        UNet = UNet()
        UNet.evaluate()
    elif architecture == '--UNet++':
        UNetPP = UNetPP()
        UNetPP.evaluate()
    else:
        raise Exception("You have passed an invalid argument.\nCheck the documentation for the allowed arguments.")

else:
    raise Exception("You have passed an invalid argument.\nCheck the documentation for the allowed arguments.")
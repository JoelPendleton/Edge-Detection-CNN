import sys
from Models.UNet import UNet
from Models.AutoEncoder1 import AutoEncoder1
from Models.AutoEncoder2 import AutoEncoder2
from Models.UNetPP import UNetPP


try:
    command = sys.argv[1]
except IndexError:
    print("An Index error has occured! Check the documentation to make sure your passing all the required arguments.")

if command == '--help':
    print("Check the documentation.")

elif command == "--train" or command == "--predict":
    try:
        architecture = sys.argv[2]
    except IndexError:
        print("An Index error has occured! Check the documentation to make sure your passing all the required arguments.")

    if command == "--train":
        if architecture == '--UNet':
            UNet = UNet()
            UNet.train()
        elif architecture == '--AutoEncoder1':
            AutoEncoder1 = AutoEncoder1()
            AutoEncoder1.train()
        elif architecture == '--AutoEncoder2':
            AutoEncoder2 = AutoEncoder2()
            AutoEncoder2.train()
        elif architecture == '--UNet++':
            UNetPP = UNetPP()
            UNetPP.train()
        else:
            raise Exception( "You have passed an invalid argument.\nCheck the documentation for the allowed arguments.")

    elif command == "--predict":
        if architecture == '--UNet':
            UNet = UNet()
            UNet.predict()
        elif architecture == '--AutoEncoder1':
            AutoEncoder1 = AutoEncoder1()
            AutoEncoder1.predict()
        elif architecture == '--AutoEncoder2':
            AutoEncoder2 = AutoEncoder2()
            AutoEncoder2.predict()
        elif architecture == '--UNet++':
            UNetPP = UNetPP()
            UNetPP.predict()
        else:
            raise Exception("You have passed an invalid argument.\nCheck the documentation for the allowed arguments.")
    else:
        raise Exception("You have passed an invalid argument.\nCheck the documentation for the allowed arguments.")


elif command == "--summary":
    architecture = sys.argv[2]

    if architecture == '--UNet':
        UNet = UNet()
        UNet.summary()
    elif architecture == '--AutoEncoder1':
        AutoEncoder1 = AutoEncoder1()
        AutoEncoder1.summary()
    elif architecture == '--AutoEncoder2':
        AutoEncoder2 = AutoEncoder2()
        AutoEncoder2.summary()
    elif architecture == '--UNet++':
        UNetPP = UNetPP()
        UNetPP.summary()
    else:
        raise Exception("You have passed an invalid argument.\nCheck the documentation for the allowed arguments.")

elif command == "--evaluate":
    architecture = sys.argv[2]

    if architecture == '--UNet':
        UNet = UNet()
        UNet.evaluate()
    elif architecture == '--AutoEncoder1':
        AutoEncoder1 = AutoEncoder1()
        AutoEncoder1.evaluate()
    elif architecture == '--AutoEncoder2':
        AutoEncoder2 = AutoEncoder2()
        AutoEncoder2.evaluate()
    elif architecture == '--UNet++':
        UNetPP = UNetPP()
        UNetPP.evaluate()
    else:
        raise Exception("You have passed an invalid argument.\nCheck the documentation for the allowed arguments.")

else:
    raise Exception("You have passed an invalid argument.\nCheck the documentation for the allowed arguments.")
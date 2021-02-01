# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import importlib, sys, os
from config import Config
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from module.enhance.rotation45 import ImageEnhance
from module.dataset import Dataset
from module.color import bcolors

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print( bcolors.WARNING + "Warning: The current configuration is CPU training neural network, the training speed is relatively slow." + bcolors.UNDERLINE)

    # config = Config()
    # te = Dataset(config)
    # te.createTrainImageGenerate()
    #
    # print(type(te.getTrainDataset()))
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os, gc
gc.collect()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from module.config import Config

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config = Config()

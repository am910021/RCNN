# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os, gc
gc.collect()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from config import Config
from module.datasethelper import DatasetHelper
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from module.modelhelper import ModelHelper
from module.callbackhalper import CallbackHelper


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config = Config()
    callback = CallbackHelper(config)
    model = ModelHelper(config)
    #model.load_h5_model()
    model.create_model()
    dataset = DatasetHelper(config)
    dataset.create_train_image_generate()
    traindata = dataset.get_train_dataset()
    testdata = dataset.get_test_dateset()

    hist = model.get_model().fit(
        traindata,
        batch_size=config.BATCH_SIZE,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        epochs=config.MAX_EPOCHS,
        validation_data=testdata,
        validation_steps=2,
        callbacks=callback.get_callbacks())

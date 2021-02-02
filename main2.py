# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os, gc
gc.collect()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from config import Config
from module.dataset import Dataset
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from module.modelhelper import ModelHelper



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config = Config()
    model = ModelHelper(config)
    #model.load_h5_model()
    model.create_model()
    dataset = Dataset(config)
    dataset.createTrainImageGenerate()
    traindata = dataset.getTrainDataset()
    testdata = dataset.getTestDateset()

    now = datetime.now()
    current_time = now.strftime("/%Y%m%d%H%M%S/")

    checkpoint = ModelCheckpoint(
        config.CHECKPOINT_PATH + "/" + config.CNN_MODEL_FILE + current_time + config.CHECKPOINT_MODEL,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=config.SAVE_PERIOD)

    checkpoint2 = ModelCheckpoint(
        config.CHECKPOINT_PATH + "/" + config.CNN_MODEL_FILE + current_time + config.CHECKPOINT_WEIGHTS + '.{epoch:05d}',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=config.SAVE_PERIOD)

    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=config.PATIENCE, verbose=1, mode='auto')

    hist = model.get_model().fit(
        traindata,
        batch_size=config.BATCH_SIZE,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        epochs=config.MAX_EPOCHS,
        validation_data=testdata,
        validation_steps=2,
        callbacks=[checkpoint, checkpoint2, early])

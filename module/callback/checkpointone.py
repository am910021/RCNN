from module.callback._abstract import TrainCallbackAbstract
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import Callback


class TrainCallback(TrainCallbackAbstract):
    def create_callback(self) -> ModelCheckpoint:
        return ModelCheckpoint(
            self.config.CHECKPOINT_PATH + "/" + self.config.CNN_MODEL_FILE + self.get_time_path() + self.config.CHECKPOINT_MODEL + '.{epoch:05d}',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=self.config.SAVE_PERIOD)

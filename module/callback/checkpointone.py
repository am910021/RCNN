from module.callback._abstract import TrainCallbackAbstract
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import Callback


class TrainCallback(TrainCallbackAbstract):
    def create_callback(self) -> ModelCheckpoint:
        ep_srt = '{epoch:0' + str(len(str(self.config.MAX_EPOCHS))) + 'd}.'
        return ModelCheckpoint(
            self.config.CHECKPOINT.CHECKPOINT_PATH + "/" + self.config.CNN_MODEL_FILE + self.get_time_path() + ep_srt + self.config.CHECKPOINT.CHECKPOINT_MODEL,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=self.config.SAVE_PERIOD)

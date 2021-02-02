from module.callback._abstract import TrainCallbackAbstract
from tensorflow.keras.callbacks import EarlyStopping


class TrainCallback(TrainCallbackAbstract):
    def create_callback(self) -> EarlyStopping:
        return EarlyStopping(monitor='val_loss', min_delta=0, patience=self.config.PATIENCE, verbose=1, mode='auto')

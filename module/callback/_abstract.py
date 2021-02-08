from module.config import Config
from keras.callbacks import Callback


class TrainCallbackAbstract:
    def __init__(self, config: Config):
        self.config = config

    def get_time_path(self) -> str:
        return self.config.TIME_PATH

    def create_callback(self) -> Callback:
        pass

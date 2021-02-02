from config import Config
from datetime import datetime
from keras.callbacks import Callback


class TrainCallbackAbstract:
    def __init__(self, config: Config):
        self.config = config

    def get_time_path(self) -> str:
        now = datetime.now()
        return now.strftime("/%Y%m%d%H%M%S/")

    def create_callback(self) -> Callback:
        pass

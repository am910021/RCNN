from tensorflow.python.keras.engine.functional import Functional
from module.config import Config


class NetModelInterface:
    def __init__(self, config:Config):
        self.Config = config

    def createNewModel(self)  -> Functional:
        raise Exception('NotImplementedException')
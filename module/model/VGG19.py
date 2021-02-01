import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.keras.optimizers import Adam
from config import Config
from module.model._interface import NetModelInterface


class NetModel(NetModelInterface):
    def __init__(self, config: Config):
        super().__init__(config)

    def createNewModel(self) -> Functional:
        opt = Adam(lr=self.Config.LEARNING_RATE)

        self.model_final = tf.keras.applications.VGG19(
            input_shape=(self.Config.IMG_WIDTH, self.Config.IMG_HEIGHT, self.Config.IMG_CHANNEL),
            weights=None, include_top=True,
            classifier_activation="softmax",
            classes=self.Config.CLASSIFICATION
        )

        return self.model_final
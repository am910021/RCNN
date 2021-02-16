import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional
from module.config import Config
from module.model._interface import NetModelInterface


class NetModel(NetModelInterface):
    def __init__(self, config: Config):
        super().__init__(config)

    def createNewModel(self) -> Functional:

        self.model_final = tf.keras.applications.InceptionResNetV2(
            input_shape=(self.Config.IMG_WIDTH, self.Config.IMG_HEIGHT, self.Config.IMG_CHANNEL),
            weights=None, include_top=True,
            classifier_activation="softmax",
            classes=len(self.Config.ANNO_LABELS)
        )

        return self.model_final
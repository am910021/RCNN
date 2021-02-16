from tensorflow.python.keras.engine.functional import Functional
from module.config import Config
from module.model._interface import NetModelInterface

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import numpy as np


class NetModel(NetModelInterface):
    def __init__(self, config: Config):
        super().__init__(config)

    def createNewModel(self) -> Functional:
        seed = 7
        np.random.seed(seed)

        self.model_final = Sequential()
        self.model_final.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(
            self.Config.IMG_WIDTH, self.Config.IMG_HEIGHT, self.Config.IMG_CHANNEL), padding='valid', activation='relu',
                                    kernel_initializer='random_uniform'))
        self.model_final.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model_final.add(
            Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model_final.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model_final.add(
            Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model_final.add(
            Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model_final.add(
            Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model_final.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self.model_final.add(Flatten())
        self.model_final.add(Dense(4096, activation='relu'))
        self.model_final.add(Dropout(0.5))
        self.model_final.add(Dense(4096, activation='relu'))
        self.model_final.add(Dropout(0.5))
        self.model_final.add(Dense(len(self.Config.ANNO_LABELS), activation='softmax'))

        return self.model_final

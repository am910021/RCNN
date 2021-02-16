from tensorflow.python.keras.engine.functional import Functional
from module.config import Config
from module.model._interface import NetModelInterface
from tensorflow.keras import layers
from tensorflow import keras


class NetModel(NetModelInterface):
    def __init__(self, config: Config):
        super().__init__(config)

    def createNewModel(self) -> Functional:
        model = keras.Sequential(name="AlexNet")

        # 輸入層請用 keras.Input 名稱為input_1，不然無法在純OpenCV下運作
        model.add(
            keras.Input(shape=(self.config.IMG_WIDTH, self.config.IMG_HEIGHT, self.config.IMG_CHANNEL), name="input_1"))

        model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu',
                                kernel_initializer='uniform'))

        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(
            layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(
            layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(
            layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(
            layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(len(self.config.DATASET.ANNO_LABELS), activation='softmax', name="predictions"))

        self.model_final = model
        return self.model_final

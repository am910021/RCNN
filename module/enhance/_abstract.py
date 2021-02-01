from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import Config
from tensorflow.python.keras.preprocessing.image import NumpyArrayIterator


class EnhanceAbstract:
    def __init__(self, config: Config):
        self.config = config

    # ImageDataGenerator 使用手冊 https://keras.io/api/preprocessing/image/
    def createEnhanceTrain(self, x_train, y_train) -> NumpyArrayIterator:
        originn_gen = ImageDataGenerator()
        return originn_gen.flow(x=x_train, y=y_train, batch_size=self.config.BATCH_SIZE)

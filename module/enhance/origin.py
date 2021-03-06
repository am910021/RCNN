from tensorflow.keras.preprocessing.image import ImageDataGenerator
from module.enhance._abstract import EnhanceAbstract
from tensorflow.python.keras.preprocessing.image import NumpyArrayIterator


class ImageEnhance(EnhanceAbstract):

    # ImageDataGenerator 使用手冊 https://keras.io/api/preprocessing/image/
    def createEnhanceTrain(self, x_train, y_train) -> NumpyArrayIterator:
        originn_gen = ImageDataGenerator()
        return originn_gen.flow(x=x_train, y=y_train, batch_size=self.config.BATCH_SIZE)

import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from module.enhance._abstract import EnhanceAbstract


class ImageEnhance(EnhanceAbstract):

    def createEnhanceTrain(self, x_train, y_train):
        datagen_args = dict(rotation_range=22.5,  # 角度值，0~180，影象旋轉
                            width_shift_range=0.2,  # 水平平移，相對總寬度的比例
                            height_shift_range=0.2,  # 垂直平移，相對總高度的比例
                            shear_range=0.2,  # 隨機錯切換角度
                            zoom_range=0.2,  # 隨機縮放範圍
                            horizontal_flip=True, vertical_flip=True,  # 一半影象水平翻轉
                            fill_mode='nearest')  # 填充新建立畫素的方法

        train_gen = ImageDataGenerator(**datagen_args)
        return train_gen.flow(x=x_train, y=y_train, batch_size=self.config.BATCH_SIZE)

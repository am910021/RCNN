#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, gc

gc.collect()

import sys, cv2
from module.config import Config
import importlib
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
import csv


class DatasetHelper:
    # 讀取資料集 資料前處理
    def load_origin_data(self):

        # 分類列表
        classification_list = self.config.ANNO_LABELS
        self.classification_len = len(classification_list)

        # 圖片列表
        img_list = os.listdir(self.config.IMG_PATH)
        img_count = len(img_list)

        # 讀取圖片列表
        for index, img_name in enumerate(img_list):
            img_path = os.path.join(self.config.IMG_PATH, img_name)  # 圖片路徑
            image = cv2.imread(img_path)  # 讀取圖片
            csv_name = os.path.splitext(img_name)[0] + ".csv"  # csv檔名
            imout = image.copy()

            # 讀取類別
            for cli, cl in enumerate(classification_list):

                sys.stdout.write("\rLoad Annotations file from %s, progress %d/%d                        " % (
                    csv_name, index + 1, img_count))
                sys.stdout.flush()

                # create classification list
                c = [0] * self.classification_len
                c[cli] = 1
                # load csv file
                csv_path = os.path.join(self.config.ANNOT, cl, csv_name)
                # print(csv_path)
                if not os.path.exists(csv_path):
                    continue

                with open(csv_path, newline='') as csvfile:

                    rows = csv.reader(csvfile, delimiter=',')
                    for trow in rows:
                        x, y, w, h = list(map(int, trow))
                        timage = imout[y:y + h, x:x + w]
                        resized = cv2.resize(timage, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT),
                                             interpolation=cv2.INTER_AREA)
                        self.train_labels.append(c)
                        self.train_images.append(resized)
        sys.stdout.write("\rLoad Annotations success.                                                           ")
        sys.stdout.flush()
        print()

    def create_train_image_generate(self):
        temp_train = []
        raw_events = itertools.chain()

        # 分割原始資料集 訓練集 測式集
        train_images, train_labels, valid_images, valid_labels = train_test_split(
            np.array(self.train_images),
            np.array(self.train_labels),
            test_size=self.config.VALID_DATASET_SIZE / 100
        )

        # 產生測式集資料
        originn_gen = ImageDataGenerator()
        self.__testDataset = itertools.chain(self.__testDataset,
                                             originn_gen.flow(x=train_labels, y=valid_labels, batch_size=self.config.BATCH_SIZE))

        # 讀設所有「增強學習的程式」，並執行
        for t in self.config.CHECKPOINT.IMAGE_ENHANCE_FILE:
            sys.stdout.write("\rCreating %s enhance image dataset." % t)
            sys.stdout.flush()

            # 從文字import類別
            ImageEnhance = None
            try:
                imp = importlib.import_module("module.enhance." + t)
                ImageEnhance = getattr(imp, 'ImageEnhance')
            except Exception as ex:
                print("Can't load module.enhance." + t)
                print("Process  terminated.")
                sys.exit()

            # 產生「增強學習的程式」的實例
            imageGenerate = ImageEnhance(self.config)
            # temp_train.append(imageGenerate.createEnhanceTrain(x_train, y_train))
            self.__trainDataset = itertools.chain(self.__trainDataset,
                                                  imageGenerate.createEnhanceTrain(train_images, valid_images))

            sys.stdout.write("\rEnhance %s image dataset created ." % t)
            sys.stdout.flush()
            print()

    def get_train_dataset(self):
        return self.__trainDataset

    def get_test_dateset(self):
        return self.__testDataset

    def reload(self):
        self.train_images = []
        self.train_labels = []
        self.load_origin_data()
        self.create_train_image_generate()

    def __init__(self, config: Config):
        self.config = config
        self.imageGenerate = None

        # OpenCV優化
        cv2.setUseOptimized(self.config.OPENCV.ENABLE_OPENCV_OPTIMIZED)
        # Selective Search物體偵測候選區域
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.train_images = []
        self.train_labels = []
        self.classification_len = 0
        self.load_origin_data()
        self.__trainDataset = itertools.chain()
        self.__testDataset = itertools.chain()

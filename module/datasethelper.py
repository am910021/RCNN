#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, sys, cv2
from module.config import Config
import pickle
import hashlib
import importlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
import csv


class DatasetHelper:
    # 讀取資料集 資料前處理
    def load_origin_data(self):

        # 分類列表
        classification_list = os.listdir(self.config.ANNOT)
        classification_len = len(classification_list)

        # 圖片列表
        img_list = os.listdir(self.config.PATH)
        img_count = len(img_list)

        # 讀取圖片列表
        for index, img_name in enumerate(img_list):
            img_path = os.path.join(self.config.PATH, img_name)  # 圖片路徑
            image = cv2.imread(img_path)  # 讀取圖片
            csv_name = os.path.splitext(img_name)[0] + ".csv"  # csv檔名
            imout = image.copy()

            # 讀取類別
            for cli, cl in enumerate(classification_list):

                sys.stdout.write("\rLoad Annotations file from %s, progress %d/%d                        " % (
                csv_name, index + 1, img_count))
                sys.stdout.flush()

                # create classification list
                c = [0] * classification_len
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

        if (self.save_cache):
            sys.stdout.write("\rSaving dataset to cache file.")
            sys.stdout.flush()

            img_hash = hashlib.sha256(repr(self.train_images).encode()).hexdigest()
            lab_hash = hashlib.sha256(repr(self.train_labels).encode()).hexdigest()

            tmp_images = [img_hash, self.train_images]
            tmp_labels = [lab_hash, self.train_labels]
            pickle.dump(tmp_images, open(self.config.DATASET_IMAGES_CACHE_NAME, "wb"))
            pickle.dump(tmp_labels, open(self.config.DATASET_LABELS_CACHE_NAME, "wb"))

            sys.stdout.write("\rDataset cache saved. ")
            sys.stdout.flush()
        print()

    def load_cache_Or_load_data(self):
        if (self.config.LOAD_CACHE_DATASET
                and os.path.exists(self.config.DATASET_IMAGES_CACHE_NAME)
                and os.path.exists(self.config.DATASET_LABELS_CACHE_NAME)):
            sys.stdout.write("\rDataset cache loading.")
            sys.stdout.flush()

            loaded = False
            try:
                load_images = pickle.load(open(self.config.DATASET_IMAGES_CACHE_NAME, "rb"))
                load_labels = pickle.load(open(self.config.DATASET_LABELS_CACHE_NAME, "rb"))

                if (type(load_images) is list and type(load_labels) is list):
                    temp_images = load_images[1]
                    temp_labels = load_labels[1]
                    img_hash = hashlib.sha256(repr(temp_images).encode()).hexdigest()
                    lab_hash = hashlib.sha256(repr(temp_labels).encode()).hexdigest()

                    if (img_hash == load_images[0] and lab_hash == load_labels[0]):
                        self.train_images = temp_images
                        self.train_labels = temp_labels
                        loaded = True

            except Exception as ex:
                loaded = False

            if (not loaded):
                sys.stdout.write("\rDetected dataset cache broken, reload dataset.")
                sys.stdout.flush()
                self.load_origin_data()
            else:
                sys.stdout.write("\rLoad dataset cache success. ")
                sys.stdout.flush()
        else:
            self.load_origin_data()
        print()

    def create_train_image_generate(self):
        temp_train = []
        raw_events = itertools.chain()

        # 分割原始資料集 訓練集 測式集
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(self.train_images),
            np.array(self.train_labels),
            test_size=self.config.TEST_DATASET_SIZE / 100
        )

        # 產生測式集資料
        originn_gen = ImageDataGenerator()
        self.__testDataset = itertools.chain(self.__testDataset,
                                             originn_gen.flow(x=x_test, y=y_test, batch_size=self.config.BATCH_SIZE))

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
                                                  imageGenerate.createEnhanceTrain(x_train, y_train))

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

    def __init__(self, config: Config, save_cache=True):
        self.config = config
        self.save_cache = save_cache
        self.imageGenerate = None

        # OpenCV優化
        cv2.setUseOptimized(self.config.OPENCV.ENABLE_OPENCV_OPTIMIZED)
        # Selective Search物體偵測候選區域
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.train_images = []
        self.train_labels = []
        self.load_cache_Or_load_data()
        self.__trainDataset = itertools.chain()
        self.__testDataset = itertools.chain()

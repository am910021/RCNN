#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, sys, cv2
from config import Config
import pickle
import hashlib
import pandas as pd
import importlib
from module.iou import IOU
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools


class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


class DatasetHelper:
    # 讀取資料集 資料前處理
    def load_origin_data(self, saveCache=True):
        annot_list = os.listdir(self.config.ANNOT)
        total_amount = len(annot_list)
        print("Loading dataset.")

        for index, file in enumerate(annot_list):
            sys.stdout.write("\rProgress %d/%d" % (index + 1, total_amount))
            sys.stdout.flush()

            try:
                if file.startswith(self.config.FILE_PREFIXES):
                    filename = file.split(".")[0] + ".jpg"
                    # print(index,filename)
                    image = cv2.imread(os.path.join(self.config.PATH, filename))
                    df = pd.read_csv(os.path.join(self.config.ANNOT, file))
                    gtvalues = []
                    for row in df.iterrows():
                        x1 = int(row[1][0].split(" ")[0])
                        y1 = int(row[1][0].split(" ")[1])
                        x2 = int(row[1][0].split(" ")[2])
                        y2 = int(row[1][0].split(" ")[3])
                        gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
                    self.ss.setBaseImage(image)
                    self.ss.switchToSelectiveSearchFast()
                    ssresults = self.ss.process()
                    imout = image.copy()
                    counter = 0
                    falsecounter = 0
                    flag = 0
                    fflag = 0
                    bflag = 0
                    for index, result in enumerate(ssresults):
                        if index < 2000 and flag == 0:
                            for gtval in gtvalues:
                                x, y, w, h = result
                                iou = IOU.calc(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                                if counter < 30:
                                    if iou > 0.70:
                                        timage = imout[y:y + h, x:x + w]
                                        resized = cv2.resize(timage, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT),
                                                             interpolation=cv2.INTER_AREA)
                                        self.train_images.append(resized)
                                        self.train_labels.append(1)
                                        counter += 1
                                else:
                                    fflag = 1
                                if falsecounter < 30:
                                    if iou < 0.3:
                                        timage = imout[y:y + h, x:x + w]
                                        resized = cv2.resize(timage, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT),
                                                             interpolation=cv2.INTER_AREA)
                                        self.train_images.append(resized)
                                        self.train_labels.append(0)
                                        falsecounter += 1
                                else:
                                    bflag = 1
                            if fflag == 1 and bflag == 1:
                                # print("inside")
                                flag = 1
            except Exception as ex:
                print(ex)
                print("error in " + filename)
                continue
        sys.stdout.write("\rLoad complete.     ")
        sys.stdout.flush()
        print()

        if (saveCache):
            sys.stdout.write("\rSaving dataset cache.")
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
                sys.stdout.write("\rLoad dataset cache complete. ")
                sys.stdout.flush()
        else:
            self.load_origin_data()
        print()

    def create_train_image_generate(self):
        temp_train = []
        raw_events = itertools.chain()

        # 分割原始資料集 訓練集 測式集
        lenc = MyLabelBinarizer()
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(self.train_images),
            lenc.fit_transform(np.array(self.train_labels)),
            test_size=self.config.TEST_DATASET_SIZE / 100
        )

        # 產生測式集資料
        originn_gen = ImageDataGenerator()
        self.__testDataset = itertools.chain(self.__testDataset,
                                             originn_gen.flow(x=x_test, y=y_test, batch_size=self.config.BATCH_SIZE))

        # 判斷是否有設定「增強學習的程式」
        if len(self.config.IMAGE_ENHANCE_FILE) == 0:
            print("Please config IMAGE_ENHANCE_FILE, then restart program.")
            print("Process  terminated.")
            sys.exit()

        # 讀設所有「增強學習的程式」，並執行
        for t in self.config.IMAGE_ENHANCE_FILE:
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

    def __init__(self, config: Config):
        self.config = config
        self.imageGenerate = None

        # OpenCV優化
        cv2.setUseOptimized(self.config.ENABLE_OPENCV_OPTIMIZED)
        # Selective Search物體偵測候選區域
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.train_images = []
        self.train_labels = []
        self.load_cache_Or_load_data()
        self.__trainDataset = itertools.chain()
        self.__testDataset = itertools.chain()
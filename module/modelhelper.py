# coding: utf-8

from module.config import Config
from tensorflow.python.keras.engine.functional import Functional
import tensorflow as tf
import importlib
import sys, time, os
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib
from os import path
from module.color import bcolors
from module.datasethelper import DatasetHelper
from module.callbackhalper import CallbackHelper
import matplotlib.pyplot as plt


class ModelHelper:
    def __init__(self, config: Config):
        self.config = config
        self.__model_final = None
        self.__init_model()

    def __get_available_gpus(self):
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def __get_final_devices(self) -> list:
        devices = []
        if len(self.config.USE_GPU_LIST) == 0:
            devices = self.__get_available_gpus()
        elif len(self.config.USE_GPU_LIST) > 0:
            devices = self.config.USE_GPU_LIST
        return devices

    def __create_model(self, printSummary=False):

        sys.stdout.write("\rCreating model from %s ." % self.config.CNN_MODEL_FILE)
        sys.stdout.flush()

        # 讀取CNN類別
        NetModel = None
        try:
            imp = importlib.import_module("module.model." + self.config.CNN_MODEL_FILE)
            NetModel = getattr(imp, 'NetModel')
        except Exception as ex:
            print("Can't load module.model." + self.config.CNN_MODEL_FILE)
            print("Process  terminated.")
            sys.exit()
        # 新建類別
        cnnModel = NetModel(self.config)

        # 指定運算設備 true=gpu  false=cpu
        if self.config.ENABLE_GPU_ACCELERATE:
            mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.__get_final_devices())
            with mirrored_strategy.scope():
                opt = Adam(lr=self.config.LEARNING_RATE)
                cnnModel = NetModel(self.config)
                self.__model_final = cnnModel.createNewModel()
                self.__model_final.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
        else:
            print()
            print(bcolors.WARNING + "Warning: The current configuration is CPU mode." + bcolors.ENDC)
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            opt = Adam(lr=self.config.LEARNING_RATE)
            cnnModel = NetModel(self.config)
            self.__model_final = cnnModel.createNewModel()
            self.__model_final.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt,
                                       metrics=["accuracy"])
        if printSummary:
            self.__model_final.summary()


        print()
        print('From ' + self.config.CNN_MODEL_FILE + ' create model success.')
        time.sleep(5)

    def __load_h5_model(self, printSummary=False):
        if not path.exists(self.config.LOAD_CHECKPOINT_H5_MODEL):
            print("Unable to load the checkpoint model at " + self.config.LOAD_CHECKPOINT_H5_MODEL)
            print("Process  terminated.")
            sys.exit()

        sys.stdout.write("\rLoading saved model from %s ." % self.config.LOAD_CHECKPOINT_H5_MODEL)
        sys.stdout.flush()

        # 指定運算設備 true=gpu  false=cpu
        if self.config.ENABLE_GPU_ACCELERATE:
            mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.__get_final_devices())
            with mirrored_strategy.scope():
                self.__model_final = tf.keras.models.load_model(self.config.LOAD_CHECKPOINT_H5_MODEL)
        else:
            print(bcolors.WARNING + "Warning: The current configuration is CPU mode." + bcolors.ENDC)
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.__model_final = tf.keras.models.load_model(self.config.LOAD_CHECKPOINT_H5_MODEL)

        if printSummary:
                self.__model_final.summary()
        print('\rFrom ' + self.config.LOAD_CHECKPOINT_H5_MODEL + ' load model success.')
        time.sleep(5)

    def get_model(self) -> Functional:
        return self.__model_final

    def __init_model(self):
        if self.config.ENABLE_LOAD_CHECKPOINT_MODEL:
            self.__load_h5_model()
        else:
            self.__create_model()

    def run_train_cnn(self, dataset: DatasetHelper, callback: CallbackHelper):
        hist = self.__model_final.fit(
            dataset.get_train_dataset(),
            batch_size=self.config.BATCH_SIZE,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            epochs=self.config.MAX_EPOCHS,
            validation_data=dataset.get_test_dateset(),
            validation_steps=2,
            callbacks=callback.get_callbacks())

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title("model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Loss", "Validation Loss"])
        plt.savefig(
            self.config.CHECKPOINT_PATH + "/" + self.config.CNN_MODEL_FILE + self.config.TIME_PATH + 'chart loss.png')
        print("train end, loss chart save in " + self.config.CHECKPOINT_PATH + "/" + self.config.CNN_MODEL_FILE + self.config.TIME_PATH + 'chart loss.png')

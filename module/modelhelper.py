# coding: utf-8

from module.config import Config
from tensorflow.python.keras.engine.functional import Functional
import tensorflow as tf
import importlib
import sys, time
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib
from os import path
from module.color import bcolors
from module.datasethelper import DatasetHelper
from module.callbackhalper import CallbackHelper


class ModelHelper:
    def __init__(self, config: Config):
        self.config = config
        self.model_final = None
        self.__init_model()

    def __get_available_gpus(self):
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def __get_final_devices(self) -> list:
        devices = ['/device:CPU:0']
        if self.config.ENABLE_GPU_ACCELERATE and len(self.config.USE_GPU_LIST) == 0:
            devices = self.__get_available_gpus()
        elif self.config.ENABLE_GPU_ACCELERATE and len(self.config.USE_GPU_LIST) > 0:
            devices = self.config.USE_GPU_LIST

        if any("CPU" in s for s in devices):
            print(
                bcolors.WARNING + "Warning: The current configuration is CPU training neural network, the training speed is relatively slow." + bcolors.ENDC)
        time.sleep(5)
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

        # 指定運算設備
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.__get_final_devices())
        with mirrored_strategy.scope():
            opt = Adam(lr=self.config.LEARNING_RATE)
            cnnModel = NetModel(self.config)
            self.model_final = cnnModel.createNewModel()
            self.model_final.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
            if printSummary:
                self.model_final.summary()
        print()
        print('From ' + self.config.CNN_MODEL_FILE + ' create model success.')
        time.sleep(5)

    def __load_h5_model(self, printSummary=False):
        if not path.exists(self.config.LOAD_CHECKPOINT_H5_FILE):
            print("Unable to load the checkpoint model at " + self.config.LOAD_CHECKPOINT_H5_FILE)
            print("Process  terminated.")
            sys.exit()

        sys.stdout.write("\rLoading saved model from %s ." % self.config.LOAD_CHECKPOINT_H5_FILE)
        sys.stdout.flush()

        # 指定運算設備
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.__get_final_devices())
        with mirrored_strategy.scope():
            self.model_final = tf.keras.models.load_model(self.config.LOAD_CHECKPOINT_H5_FILE)
            if printSummary:
                self.model_final.summary()
        print('\rFrom ' + self.config.LOAD_CHECKPOINT_H5_FILE + ' load model success.')
        time.sleep(5)

    def get_model(self) -> Functional:
        return self.model_final

    def __init_model(self):
        if self.config.ENABLE_LOAD_CHECKPOINT_H5:
            self.__load_h5_model()
        else:
            self.__create_model()

    def run_train_cnn(self, dataset: DatasetHelper, callback: CallbackHelper):
        self.model_final.fit(
            dataset.get_train_dataset(),
            batch_size=self.config.BATCH_SIZE,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            epochs=self.config.MAX_EPOCHS,
            validation_data=dataset.get_test_dateset(),
            validation_steps=2,
            callbacks=callback.get_callbacks())

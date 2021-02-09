# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os, gc, sys, cv2

gc.collect()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from module.config import Config
from module.modelhelper import ModelHelper
from os import path
import numpy as np
import threading, time
from threading import Lock
from tensorflow.python.keras.engine.functional import Functional
from numpy import ndarray

class Selective:
    def __init__(self, config: Config, file, ss):
        self.__img = cv2.imread(os.path.join(config.DETECTOR.INPUT_PATH, file))
        ss.setBaseImage(self.__img)
        ss.switchToSelectiveSearchFast()
        self.__ssresults = ss.process()
        self.__imout = self.__img.copy()
        self.__count = 0
        self.__read_lock = Lock()
        self.__write_lock = Lock()
        self.__len = len(self.__ssresults)
        self.file = file

    def get_selective(self):
        self.__read_lock.acquire()
        ret = self.__ssresults[self.__count]
        self.__count += 1
        sys.stdout.write("\rThe %s region processing progress is %d/%d." % (self.file, self.__count, self.__len))
        sys.stdout.flush()
        self.__read_lock.release()
        return ret

    def is_end(self) -> bool:
        return self.__count >= len(self.__ssresults)-1

    def write(self, x, y, w, h):
        self.__read_lock.acquire()
        cv2.rectangle(self.__imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        self.__read_lock.release()

    def get_imout(self):
        return self.__imout

    def get_img(self):
        return self.__img


class Thread(threading.Thread):
    def __init__(self, model: Functional, selective: Selective):
        threading.Thread.__init__(self)
        self.selective = selective
        self.model = model

    def run(self):
        while not self.selective.is_end():
            x, y, w, h = self.selective.get_selective()
            timage = self.selective.get_img()[y:y + h, x:x + w]
            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)
            out = self.model.predict(img)
            if out[0][0] > 0.85:
                self.selective.write(x, y, w, h)


def detector(config: Config, model: Functional):
    if not path.exists(config.DETECTOR.INPUT_PATH):
        print(config.DETECTOR.INPUT_PATH + "not exists.")
        sys.exit()

    if not path.exists(config.DETECTOR.OUTPUT_PATH):
        os.makedirs(config.DETECTOR.OUTPUT_PATH)

    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    for file in os.listdir(config.DETECTOR.INPUT_PATH):
        selective = Selective(config, file, ss)
        thread = []

        for i in range(0, config.DETECTOR.WORKERS):
            temp = Thread(model, selective)
            temp.start()
            thread.append(temp)

        for i in thread:
            i.join()

        cv2.imwrite(path.join(config.DETECTOR.OUTPUT_PATH, file), selective.get_imout())
        print("\r%s Done." % file)
        print()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config = Config()
    model = ModelHelper(config)
    detector(config, model.get_model())

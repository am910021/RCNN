import os

import numpy as np
import cv2, sys
from os import path
from module.config import Config

def detector(ss, net, config):

    for file in os.listdir(config.DETECTOR.INPUT_PATH):

        img = cv2.imread(path.join(config.DETECTOR.INPUT_PATH, file))
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = img.copy()

        max_ss = min(len(ssresults), 2000)
        for index in range(0, max_ss):
            x, y, w, h = ssresults[index]
            timage = imout[y:y + h, x:x + w]

            pic = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
            blob = cv2.dnn.blobFromImage(pic,
                                         scalefactor=1,
                                         size=(224, 224),
                                         mean=(0, 0, 0),
                                         swapRB=True,
                                         crop=False)
            # blob = np.transpose(blob, (0,2,3,1))
            net.setInput(blob)
            out = net.forward()
            out = out.flatten()
            classId = np.argmax(out)
            if out[0] > 0.85:
                cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

            sys.stdout.write("\rThe %s region processing progress is %d/%d." % (file, index+1, max_ss))
            sys.stdout.flush()

        cv2.imwrite(path.join(config.DETECTOR.OUTPUT_PATH, file), imout)
        print("\r%s Done." % file)

if __name__ == '__main__':
    config = Config()
    config.init_detector_config() #載入config detector設定

    cv_model_path = path.join(config.OPENCV.CONVERT_MODE_SAVE_PATH, config.CNN_MODEL_FILE) + '.pb'
    net = cv2.dnn.readNet(cv_model_path)

    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    detector(ss, net, config)
import sys
from os import path
from configparser import ConfigParser, ExtendedInterpolation
import argparse
from datetime import datetime


class Config:

    def __create_new_config(self, name) -> bool:
        name = name.replace(".cfg.ini", "")
        if not path.exists(name + ".cfg.ini"):
            f = open("module/config.default", "r")
            tmp = f.read()
            f.close()

            f = open(name + ".cfg.ini", "w")
            f.write(tmp)
            f.close()

            print("New config created, name is '%s' ." % (name + ".cfg.ini"))
            return True
        return False

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config",
                            help="Command '-c xxx' ,Choose config file, default file is 'default.cfg.ini'",
                            type=str)
        parser.add_argument("-n", "--new", help="Create new config file.", type=str)
        args = parser.parse_args()

        is_new = True if args.new else False
        config_file = args.config.replace(".cfg.ini", "") + ".cfg.ini" if args.config else None

        if (is_new):
            if not self.__create_new_config(args.new):
                print("file name '%s' exist, please choose other name." % args.new)
                sys.exit()
            else:
                print('Please re-start program and choose new config file.')
                sys.exit()

        default = 'default.cfg.ini'
        file = default
        if (config_file is None or not path.exists(config_file)) and not path.exists(default):
            print("Oops, can't find any config file.")
            self.__create_new_config(default)
            print('Please re-start program and choose new config file.')
            sys.exit()
        elif config_file and path.exists(config_file):
            file = config_file
            print("Loading config file from '%s' ." % config_file)
        else:
            print("Loading config file from '%s' ." % default)

        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(file)

        try:
            self.IMG_WIDTH = int(config['GENERAL']['IMG_WIDTH'])
            self.IMG_HEIGHT = int(config['GENERAL']['IMG_HEIGHT'])
            self.IMG_CHANNEL = int(config['GENERAL']['IMG_CHANNEL'])
            self.CLASSIFICATION = int(config['GENERAL']['CLASSIFICATION'])
            self.CNN_MODEL_FILE \
                = config['GENERAL']['CNN_MODEL_FILE'].replace('"', '').replace("'", '').replace(" ", '')
            self.CNN_TRAIN_CALLBACK \
                = config['GENERAL']['CNN_TRAIN_CALLBACK'].replace('"', '').replace("'", '').replace(" ", '').split(",")
            self.ENABLE_GPU_ACCELERATE = config['GENERAL']['ENABLE_GPU_ACCELERATE'].upper() == "TRUE"
            self.USE_GPU_LIST \
                = config['GENERAL']['USE_GPU_LIST'].replace('"', '').replace("'", '').replace(" ", '').split(",")
            self.MAX_EPOCHS = int(config['GENERAL']['MAX_EPOCHS'])
            self.BATCH_SIZE = int(config['GENERAL']['BATCH_SIZE'])
            self.STEPS_PER_EPOCH = int(config['GENERAL']['STEPS_PER_EPOCH'])
            self.SAVE_PERIOD = int(config['GENERAL']['SAVE_PERIOD'])
            self.PATIENCE = int(config['GENERAL']['PATIENCE'])
            self.LEARNING_RATE = float(config['GENERAL']['LEARNING_RATE'])

            self.DATASET_IMAGES_CACHE_NAME \
                = config['DATASET']['dataset_images_cache_name'].replace('"', '').replace("'", '').replace(" ", '')
            self.DATASET_LABELS_CACHE_NAME \
                = config['DATASET']['dataset_labels_cache_name'].replace('"', '').replace("'", '').replace(" ", '')
            self.TEST_DATASET_SIZE = int(config['DATASET']['test_dataset_size'])
            self.LOAD_CACHE_DATASET = config['DATASET']['load_cache_dataset'].upper() == "TRUE"
            self.PATH = config['DATASET']['path'].replace('"', '').replace("'", '').replace(" ", '')
            self.ANNOT = config['DATASET']['annot'].replace('"', '').replace("'", '').replace(" ", '')
            self.FILE_PREFIXES \
                = config['DATASET']['file_prefixes'].replace('"', '').replace("'", '').replace(" ", '')

            self.CHECKPOINT_PATH \
                = config['CHECKPOINT']['checkpoint_path'].replace('"', '').replace("'", '').replace(" ", '')
            self.CHECKPOINT_MODEL \
                = config['CHECKPOINT']['checkpoint_model'].replace('"', '').replace("'", '').replace(" ", '')
            self.CHECKPOINT_WEIGHTS \
                = config['CHECKPOINT']['checkpoint_weights'].replace('"', '').replace("'", '').replace(" ", '')
            self.ENABLE_LOAD_CHECKPOINT_MODEL = config['CHECKPOINT']['enable_load_checkpoint_model'].upper() == "TRUE"
            self.LOAD_CHECKPOINT_H5_FILE \
                = config['CHECKPOINT']['load_checkpoint_h5_file'].replace('"', '').replace("'", '').replace(" ", '')
            self.IMAGE_ENHANCE_FILE \
                = config['CHECKPOINT']['image_enhance_file'].replace('"', '').replace("'", '').replace(" ", '').split(
                ",")

            self.ENABLE_OPENCV_OPTIMIZED = config['OTHER']['enable_opencv_optimized'].upper() == "TRUE"
            self.CPU_THREAD = int(config['OTHER']['cpu_thread'])

            now = datetime.now()
            self.TIME_PATH = now.strftime("/%Y-%m-%d-%H-%M-%S/")

        except Exception as ex:
            print(ex)
            sys.exit()

        self.check_config_file()

    def check_config_file(self):
        sys.stdout.write("\rConfig file Initializing.")
        sys.stdout.flush()

        # 判斷是否有設定「增強學習的程式」
        if len(self.IMAGE_ENHANCE_FILE) == 0:
            print("Please configure IMAGE_ENHANCE_FILE, then restart program.")
            print("Process  terminated.")
            sys.exit()

        for file in self.IMAGE_ENHANCE_FILE:
            if not path.exists('module/enhance/' + file + '.py'):
                print("Please check module/enhance/%s.py file exists or re-configure IMAGE_ENHANCE_FILE." % file)
                print("Process  terminated.")
                sys.exit()

        if not path.exists('module/model/' + self.CNN_MODEL_FILE + '.py'):
            print(
                "Please check module/model/%s.py file exists or re-configure CNN_MODEL_FILE." % self.CNN_MODEL_FILE)
            print("Process  terminated.")
            sys.exit()

        if len(self.CNN_TRAIN_CALLBACK) == 0:
            print("Please configure CNN_TRAIN_CALLBACK, then restart program.")
            print("Process  terminated.")
            sys.exit()

        for file in self.CNN_TRAIN_CALLBACK:
            if not path.exists('module/callback/' + file + '.py'):
                print("Please check module/callback/%s.py file exists or re-configure CNN_TRAIN_CALLBACK." % file)
                print("Process  terminated.")
                sys.exit()

        if self.ENABLE_LOAD_CHECKPOINT_MODEL:
            if not path.exists(self.LOAD_CHECKPOINT_H5_FILE):
                print(
                    "Please check %s file exists or re-configure LOAD_CHECKPOINT_H5_FILE." % self.LOAD_CHECKPOINT_H5_FILE)
                print("Process  terminated.")
                sys.exit()

        sys.stdout.write("\rConfig file check success.")
        sys.stdout.flush()
        print()

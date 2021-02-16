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
        self.__config_file = config

        try:
            self.IMG_WIDTH = int(config['GENERAL']['img_width'])
            self.IMG_HEIGHT = int(config['GENERAL']['img_height'])
            self.IMG_CHANNEL = int(config['GENERAL']['img_channel'])

            self.CNN_MODEL_FILE \
                = config['GENERAL']['cnn_model_file'].replace('"', '').replace("'", '').replace(" ", '')
            self.CNN_TRAIN_CALLBACK \
                = config['GENERAL']['cnn_train_callback'].replace('"', '').replace("'", '').replace(" ", '').split(",")
            self.ENABLE_GPU_ACCELERATE = config['GENERAL']['enable_gpu_accelerate'].upper() == "TRUE"
            self.USE_GPU_LIST \
                = config['GENERAL']['use_gpu_list'].replace('"', '').replace("'", '').replace(" ", '').split(",")
            self.MAX_EPOCHS = int(config['GENERAL']['max_epochs'])
            self.BATCH_SIZE = int(config['GENERAL']['batch_size'])
            self.STEPS_PER_EPOCH = int(config['GENERAL']['steps_per_epoch'])
            self.SAVE_PERIOD = int(config['GENERAL']['save_period'])
            self.PATIENCE = int(config['GENERAL']['patience'])
            self.LEARNING_RATE = float(config['GENERAL']['learning_rate'])

            now = datetime.now()
            self.TIME_PATH = now.strftime("/%Y-%m-%d-%H-%M-%S/")
            self.TIME = now.strftime("%Y-%m-%d-%H-%M-%S")

            self.OPENCV = OPENCV(config)
            self.CHECKPOINT = CHECKPOINT(config)
            self.DATASET = DATASET(self.__config_file)

        except Exception as ex:
            print("configure check fail, please check config file or create new config.")
            print("%s key not found." % ex)
            sys.exit()

        self.check_config_file()

    def check_config_file(self):
        sys.stdout.write("\rConfig file Initializing.")
        sys.stdout.flush()

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

        if self.CHECKPOINT.ENABLE_LOAD_CHECKPOINT_H5:
            if not path.exists(self.CHECKPOINT.LOAD_CHECKPOINT_H5_MODEL):
                print(
                    "Please check %s file exists or re-configure LOAD_CHECKPOINT_H5_MODEL." % self.CHECKPOINT.LOAD_CHECKPOINT_H5_MODEL)
                print("Process  terminated.")
                sys.exit()

        sys.stdout.write("\rConfig file check success.")
        sys.stdout.flush()
        print()

    def init_detector_config(self):
        self.DETECTOR = DETECTOR(self, self.__config_file)


class DETECTOR:
    def __init__(self, config: Config, config_file: dict):
        self.config = config
        try:
            self.INPUT_PATH = config_file['DETECTOR']['input_path'].replace('"', '').replace("'", '').replace(" ", '')
            self.OUTPUT_PATH = config_file['DETECTOR']['output_path'].replace('"', '').replace("'", '').replace(" ", '')
            self.WORKERS = int(config_file['DETECTOR']['workers'].replace('"', '').replace("'", '').replace(" ", ''))
            self.DETECT_TARGET = config_file['DETECTOR']['detect_target'].replace('"', '').replace("'", '').replace(" ",
                                                                                                                    '')
            self.REQUIRE_ACCURACY = float(config_file['DETECTOR']['require_accuracy'])
            self.FAST_SELECTIVE_SEARCH = config_file['DETECTOR']['fast_selective_search'].upper() == "TRUE"

        except Exception as ex:
            print("configure check fail, please check config file or create new config.")
            print("['DETECTOR'] %s key not found." % ex)
            sys.exit()
        self.__check__()

    def __check__(self):
        sys.stdout.write("\rConfig [DETECTOR] Initializing.")
        sys.stdout.flush()

        # 判斷是否有設定「增強學習的程式」
        if self.DETECT_TARGET not in self.config.DATASET.ANNO_LABELS:
            sys.stdout.write("\rConfig [DETECTOR] check fail.")
            sys.stdout.flush()
            print()
            print("\rPlease check configure [DETECTOR] detect_target, then restart program.")
            print("\rProcess  terminated.")
            sys.exit()

        sys.stdout.write("\rConfig [DETECTOR] check success.")
        sys.stdout.flush()
        print()


class OPENCV:
    def __init__(self, config: dict):
        try:
            self.ENABLE_OPENCV_OPTIMIZED = config['OPENCV']['enable_opencv_optimized'].upper() == "TRUE"
            self.CONVERT_MODE_SAVE_PATH = config['OPENCV']['convert_mode_save_path'].replace('"', '').replace("'",
                                                                                                              '').replace(
                " ", '')
        except Exception as ex:
            print("configure check fail, please check config file or create new config.")
            print("[OPENCV] %s key not found." % ex)
            sys.exit()


class CHECKPOINT:
    def __init__(self, config: dict):
        try:
            self.CHECKPOINT_PATH \
                = config['CHECKPOINT']['checkpoint_path'].replace('"', '').replace("'", '').replace(" ", '')
            self.CHECKPOINT_MODEL \
                = config['CHECKPOINT']['checkpoint_model'].replace('"', '').replace("'", '').replace(" ", '')
            self.CHECKPOINT_WEIGHTS \
                = config['CHECKPOINT']['checkpoint_weights'].replace('"', '').replace("'", '').replace(" ", '')

            self.ENABLE_LOAD_CHECKPOINT_H5 = config['CHECKPOINT']['enable_load_checkpoint_h5'].upper() == "TRUE"
            self.LOAD_CHECKPOINT_H5_MODEL \
                = config['CHECKPOINT']['load_checkpoint_h5_model'].replace('"', '').replace("'", '').replace(" ", '')
        except Exception as ex:
            print("configure check fail, please check config file or create new config.")
            print("[CHECKPOINT] %s key not found." % ex)
            sys.exit()


class DATASET:
    def __init__(self, config: dict):
        try:
            self.VALID_DATASET_SIZE = float(config['DATASET']['valid_dataset_size']) / 100
            self.IMG_PATH = config['DATASET']['img_path'].replace('"', '').replace("'", '').replace(" ", '')
            self.ANNOT = config['DATASET']['annot'].replace('"', '').replace("'", '').replace(" ", '')
            self.ANNO_LABELS \
                = config['DATASET']['annot_labels'].replace('"', '').replace("'", '').replace(" ", '').split(",")

            self.IMAGE_ENHANCE_FILE \
                = config['DATASET']['image_enhance_file'].replace('"', '').replace("'", '').replace(" ", '').split(
                ",")
        except Exception as ex:
            print("configure check fail, please check config file or create new config.")
            print("[DATASET] %s key not found." % ex)
            sys.exit()

        self.__check__()

    def __check__(self):
        # 判斷是否有設定「增強學習的程式」
        if self.VALID_DATASET_SIZE < 0 or self.VALID_DATASET_SIZE > 1:
            print("Please configure [DATASET] valid_dataset_size, then restart program.")
            print("Process  terminated.")
            sys.exit()

        if self.IMG_PATH is "":
            print("Please configure [DATASET] img_path, then restart program.")
            print("Process  terminated.")
            sys.exit()

        if self.ANNOT is "":
            print("Please configure [DATASET] annot, then restart program.")
            print("Process  terminated.")
            sys.exit()

        if len(self.IMAGE_ENHANCE_FILE) == 0:
            print("Please configure[DATASET] image_enhance_file, then restart program.")
            print("Process  terminated.")
            sys.exit()

        for file in self.IMAGE_ENHANCE_FILE:
            if not path.exists('module/enhance/' + file + '.py'):
                print(
                    "Please check module/enhance/%s.py file exists or re-configure [DATASET] image_enhance_file." % file)
                print("Process  terminated.")
                sys.exit()

        sys.stdout.write("\rConfig [DATASET] check success.")
        sys.stdout.flush()
        print()
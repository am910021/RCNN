import sys
from os import path


class Config:
    IMG_WIDTH = 224  # 設定圖片寬
    IMG_HEIGHT = 224  # 設定圖片長
    IMG_CHANNEL = 3  # 設定圖片通道數
    CLASSIFICATION = 2  # 分類器數量

    TEST_DATASET_SIZE = 20  # 測式集大小 0~100

    # 設定增強學習的程式 位置 module/enhance/XXX.py origin=原始數據 rotation45=加入轉45角的資料
    IMAGE_ENHANCE_FILE = ['rotation45',
                          'rotation90']

    # 設定CNN模型的位置 module/model/XXX.py
    CNN_MODEL_FILE = 'InceptionResNetV2'

    # 設定callback存檔點 位置module/callback/xxx.py checkpointone=儲存模型+權重 checkpointtwo=儲存權重 earlystop=未收練結束訓練
    CNN_TRAIN_CALLBACK = ['checkpointone', 'checkpointtwo', 'earlystop']

    ENABLE_LOAD_CHECKPOINT_MODEL = False  # 載入儲存點
    LOAD_CHECKPOINT_H5_FILE = 'model.h5'  # 載入儲存點位置(h5檔)
    LOAD_CHECKPOINT_WEIGHT = 'checkpoint2/xxx'  # 載入儲存點位置(weight)

    ENABLE_GPU_ACCELERATE = True  # 使用gpu加速學習
    USE_GPU_LIST = ['/device:GPU:0', '/device:GPU:1']  # 設定GPU 全部=[]  單一GPU=['/device:GPU:0']

    MAX_EPOCHS = 10000  # 最大回合數(EPOCH)
    BATCH_SIZE = 16
    STEPS_PER_EPOCH = 10
    SAVE_PERIOD = 1
    PATIENCE = 1000  # 能夠容忍多少個回合(EPOCH)內都沒有收練(IMPROVEMENT)
    LEARNING_RATE = 0.0001  # 學習率

    CPU_THREAD = 6  # CPU線程數
    LOAD_CACHE_DATASET = True  # 使用快取資料 True=載入快取資料集 False=重新載入資料集

    PATH = "Images"  # 資料集的原始圖片位置
    ANNOT = "Airplanes_Annotations"  # 資料集的標記檔位置
    FILE_PREFIXES = "airplane"
    DATASET_IMAGES_CACHE_NAME = "train_images.cache"  # 快取檔案名稱
    DATASET_LABELS_CACHE_NAME = "train_labels.cache"  # 快取檔案名稱

    CHECKPOINT_PATH = "./checkpoint"  # 檢查點位置
    CHECKPOINT_MODEL = "model.h5"  # 檢查點檔案名稱 檢查點包含模型與權重
    CHECKPOINT_WEIGHTS = "weight"  ##檢查點權重名稱

    ENABLE_OPENCV_OPTIMIZED = True  # 啟用opencv優化

    def __init__(self):
        sys.stdout.write("\rConfig file Initializing.")
        sys.stdout.flush()

        # 判斷是否有設定「增強學習的程式」
        if len(Config.IMAGE_ENHANCE_FILE) == 0:
            print("Please configure IMAGE_ENHANCE_FILE, then restart program.")
            print("Process  terminated.")
            sys.exit()

        for file in Config.IMAGE_ENHANCE_FILE:
            if not path.exists('module/enhance/' + file + '.py'):
                print("Please check module/enhance/%s.py file exists or re-configure IMAGE_ENHANCE_FILE." % file)
                print("Process  terminated.")
                sys.exit()

        if not path.exists('module/model/' + Config.CNN_MODEL_FILE + '.py'):
            print(
                "Please check module/model/%s.py file exists or re-configure CNN_MODEL_FILE." % Config.CNN_MODEL_FILE)
            print("Process  terminated.")
            sys.exit()

        if len(Config.CNN_TRAIN_CALLBACK) == 0:
            print("Please configure CNN_TRAIN_CALLBACK, then restart program.")
            print("Process  terminated.")
            sys.exit()

        for file in Config.CNN_TRAIN_CALLBACK:
            if not path.exists('module/callback/' + file + '.py'):
                print("Please check module/callback/%s.py file exists or re-configure CNN_TRAIN_CALLBACK." % file)
                print("Process  terminated.")
                sys.exit()

        if Config.ENABLE_LOAD_CHECKPOINT_MODEL:
            if not path.exists(Config.LOAD_CHECKPOINT_H5_FILE):
                print(
                    "Please check %s file exists or re-configure LOAD_CHECKPOINT_H5_FILE." % Config.LOAD_CHECKPOINT_H5_FILE)
                print("Process  terminated.")
                sys.exit()

            if not path.exists(Config.LOAD_CHECKPOINT_WEIGHT):
                print(
                    "Please check %s file exists or re-configure LOAD_CHECKPOINT_WEIGHT." % Config.LOAD_CHECKPOINT_WEIGHT)
                print("Process  terminated.")
                sys.exit()

        sys.stdout.write("\rConfig file check success.")
        sys.stdout.flush()
        print()

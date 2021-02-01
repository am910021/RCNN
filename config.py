#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Config:
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    IMG_CHANNEL = 3
    CLASSIFICATION = 2  # 分類器數量

    TEST_DATASET_SIZE = 20  #測式集大小 0~100
    IMAGE_ENHANCE_FILE = ['origin','origin'] #設定增強學習的程式 位置 module/enhance/XXX.py origin=原始數據 rotation45=加入轉45角的資料
    CNN_MODEL_FILE = 'InceptionResNetV2'  # 設定CNN模型的位置 module/model/XXX.py

    ENABLE_LOAD_CHECKPOINT_MODEL = False  #載入儲存點
    LOAD_CHECKPOINT_H5_FILE = 'checkpoint2/full.02971.h5' # 載入儲存點位置(h5檔)
    LOAD_CHECKPOINT_WEIGHT = 'checkpoint2/xxx' # 載入儲存點位置(weight)

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

    PATH = "Images" #資料集的原始圖片
    ANNOT = "Airplanes_Annotations" #資料集的標記
    FILE_PREFIXES = "airplane"
    DATASET_IMAGES_CACHE_NAME = "train_images.cache" #快取檔案名稱
    DATASET_LABELS_CACHE_NAME = "train_labels.cache"  #快取檔案名稱

    CHECKPOINT_PATH = "./checkpoint" #檢查點位置
    CHECKPOINT_MODEL = "model.h5" #檢查點檔案名稱 檢查點包含模型與權重
    CHECKPOINT_WEIGHTS = "weight" ##檢查點權重名稱

    ENABLE_OPENCV_OPTIMIZED = True #啟用opencv優化

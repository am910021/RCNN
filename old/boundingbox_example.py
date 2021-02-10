#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, cv2
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#設定資料夾路徑
path = "Images"
annot = "Airplanes_Annotations"


# In[3]:


#印出邊界框(bounding box)示意圖
for index, file in enumerate(os.listdir(annot)):
    if index>=1:
        break
    
    filename = file.split(".")[0]+".jpg"
    img = cv2.imread(os.path.join(path,filename)) #讀取圖片
    df = pd.read_csv(os.path.join(annot,file)) #讀取邊界框資料(csv)
        
    plt.figure(edgecolor='black',linewidth=3) #產生圖片示窗
    plt.suptitle("Bounding box diagram") #圖片示窗標題
        
    plt.subplot(1,2,1)     #分割示窗 1列 2欄 第1張
    plt.title("Origin image")   #第1張圖片標題
    plt.imshow(img)      #繪製第1張圖片
        
    for row in df.iterrows():
        x1 = int(row[1][0].split(" ")[0])
        y1 = int(row[1][0].split(" ")[1])
        x2 = int(row[1][0].split(" ")[2])
        y2 = int(row[1][0].split(" ")[3])
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0), 2) #cv2.rectangle產生方框的函數
    plt.subplot(1,2,2)    #分割示窗 1列 2欄 第2張
    plt.title("Bounding box image")   #第2張圖片標題
    plt.imshow(img)      #繪製第2張圖片
        
    plt.show() #顯視圖片


# In[4]:


# Selective Search物體偵測候選區域範列
path = "Images" #圖片資料夾
cv2.setUseOptimized(True); #OpenCV優化
im = cv2.imread(os.path.join(path,"42850.jpg")) #取讀42850.jpg圖片

#手冊https://docs.opencv.org/3.4/d6/d6d/classcv_1_1ximgproc_1_1segmentation_1_1SelectiveSearchSegmentation.html
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(im)
ss.switchToSelectiveSearchFast() #快度模式
#ss.switchToSelectiveSearchQuality() #高品質模式

rects = ss.process()
imOut = im.copy()

for i, rect in (enumerate(rects)):
    x, y, w, h = rect
    cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA) #cv2.rectangle產生方框的函數

plt.figure()
plt.title("Selective Object demo")
plt.imshow(imOut)
plt.show()


# In[ ]:





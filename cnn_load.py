# import tensorflow.keras as keras
# from tensorflow.keras.models import Sequential,load_model
# from tensorflow.keras.layers import Dense,InputLayer,Dropout,Conv2D,MaxPooling2D,Flatten
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import load_model
from os import path
import os
from process_data import load_datas,img2input
import numpy as np
#import process_data
class predictor:
    def __init__(self,pth,img_siz):
        self.model=load_model(pth)
        self.img_siz=img_siz
    def predict(self,img):
        x=img2input(img,img_siz=self.img_siz,flatten=False)
        return self.model.predict(x=np.array([x]))[0]
if(__name__=='__main__'):
    
    pd=predictor(r"M:\Weiyun Sync\code\nsfw_keras\2lJ7HIOKL2qIgs7_lr=0.0055\84.80.h5",128)
    
    def get_sample(label='pos',n=5):
        from glob import glob
        import random
        ls=r'M:\Weiyun Sync\code\nsfw_keras\datas\%s'%label
        ls=list(glob(path.join(ls,'*')))
        ls=random.sample(ls,n)
        return ls
    false=0
    tot=200
    for i in get_sample(n=tot//2):
        buse,setu=pd.predict(i)
        print(i,setu>buse)
        if(buse>setu):
            false+=1
    for i in get_sample(label='neg',n=tot//2):
        buse,setu=pd.predict(i)
        print(i,setu>buse)
        if(setu>buse):
            false+=1
    print(1-false/tot)
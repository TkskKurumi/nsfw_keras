import sys
sys.path.append(r'C:\YaqianBot\yaqianok\utils')
from os import path
import os
import random,traceback
import numpy as np
import cv2
import myhash,time
from merge_files import base32
from glob import glob
from progbar import progbar
pwd=path.dirname(__file__)
data_pth=path.join(pwd,'datas')
default_exts=['jpg','gif','png','bmp']

def shashi(s,encoding='utf-8',length=64):
    bytes=s.encode(encoding)
    ret=0
    for i in bytes:
        ret=(ret<<7)|int(i)
    mask=(1<<length)-1
    while(ret>>length):
        ret=(ret&mask)^(ret>>length)
    return ret


def img2input(img,flatten=True,img_siz=224,normalize=False):
    from PIL import Image
    if(isinstance(img,str)):
        img=Image.open(img)
    img=img.resize((img_siz,img_siz),Image.LANCZOS).convert("RGB")
    if(flatten):
        if(normalize):
            return (np.asarray(img).astype(np.float32).flatten()-127.5)/127.5
            
        else:
            return np.asarray(img).astype(np.float32).flatten()
         
    else:
        if(normalize):
            return (np.asarray(img).astype(np.float32)-127.5)/127.5
        else:
            return np.asarray(img).astype(np.float32)
def glob_exts(pth,exts=None):
    global default_exts
    if(exts is None):
        exts=default_exts
    ret=[]
    for ext in exts:
        ret.extend(list(glob(path.join(pth,"*."+ext))))
    return ret
def get_data_pth(val_rate=0.05,test_rate=0.05):
    pos=glob_exts(path.join(data_pth,'pos'))
    neg=glob_exts(path.join(data_pth,'neg'))
    pth_with_label=[(i,[0,1]) for i in pos]+[(i,[1,0]) for i in neg]
    def cmpkey(x):
        pth,label=x
        return shashi(pth,length=10)
    
    pth_with_label.sort(key=cmpkey)
    
    le=len(pth_with_label)
    le_val=int(le*val_rate)
    le_test=int(le*test_rate)
    
    end_val=le_val
    end_test=end_val+le_test
    
    val=pth_with_label[:end_val]
    test=pth_with_label[end_val:end_test]
    train=pth_with_label[end_test:]
    return train,val,test

def augment(x):
    if(x.dtype!=np.uint8):
        x=x.astype(np.uint8)
    h,w=x.shape[:2]
    rot=random.random()*360
    mat=cv2.getRotationMatrix2D((random.random()*w,random.random()*h),rot,1)
    x=cv2.warpAffine(x,mat,(h,w),borderMode=cv2.BORDER_WRAP)
    
    '''x=x.astype(np.float32)+np.random.normal(loc=0,scale=8,size=x.shape).astype(np.float32)
    x=np.maximum(x,0)
    x=np.minimum(x,255)'''
    return x.astype(np.uint8)
def distribution(ys):
    mean=np.sum(ys,axis=0)/len(ys)
    neg=np.sum(ys[:,0]>0.5)
    pos=np.sum(ys[:,1]>0.5)
    return mean,neg,pos
def yield_data(mode='train',sample=None,img_size=224,save_sample=True,augment=augment,normalize=True,smooth_label=True):
    _progbar=progbar()
    train,val,test=get_data_pth()
    
    if(mode=='train'):
        pth_with_label=train
    elif(mode=='val'):
        pth_with_label=val
    else:
        pth_with_label=test
    xs=[]
    ys=[]
    if(sample):
        pth_with_label=random.sample(pth_with_label,sample)
    _label=pth_with_label[0][1]
    label_mean=sum(_label)/len(_label)
    cached_num=0
    uncache_num=0
    for idx,i in enumerate(pth_with_label):
        _progbar('loading %s data'%mode,idx,len(pth_with_label))
        pth,label=i
        h=path.basename(path.dirname(pth))+'_'+path.basename(pth)
        svname="file=%s,rot=%d,size=%d.npy"%(h,0,img_size)
        svpth=path.join('npy',svname)
        if(path.exists(svpth)):
            x=np.load(svpth)
            cached_num+=1
        else:
            try:
                x=img2input(pth,flatten=False,img_siz=img_size,normalize=False)
            except Exception as e:
                print('connot read image file %s'%pth,e)
            np.save(svpth,x)
            uncache_num+=1
        if(augment):
            
            if(save_sample):
                cv2.imwrite(path.join(pwd,'before aug.png'),cv2.cvtColor(x,cv2.COLOR_RGB2BGR))
            x=augment(x)
            if(save_sample):
                cv2.imwrite(path.join(pwd,'after aug.png'),cv2.cvtColor(x,cv2.COLOR_RGB2BGR))
                save_sample=False
        xs.append(x)
        ys.append(label)
    _progbar.finish('%d cached, %d uncached'%(cached_num,uncache_num))
    xs=np.array(xs,np.float32)
    ys=np.array(ys,np.float32)
    if(normalize):
        xs=xs/127.5-1
    if(smooth_label is not None):
        ys=ys*smooth_label+label_mean*(1-smooth_label)
    return xs,ys
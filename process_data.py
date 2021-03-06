import sys
sys.path.append(r'C:\YaqianBot\yaqianok\utils')
import keras
from keras.models import Sequential
from keras.layers import Dense
from os import path
import os
import random,traceback
import numpy as np
import myhash,time
from merge_files import base32
img_siz=48    #width=height=img_siz,RGB
n_input=img_siz*img_siz*3
n_hidden0=n_input//20
n_hidden1=int(n_hidden0/1.5)
n_classes=2    #sfw=0/nsfw=1
curpth=path.dirname(__file__)







def img2input(img,flatten=True,img_siz=img_siz,norm=False):
    from PIL import Image
    if(isinstance(img,str)):
        img=Image.open(img)
    img=img.resize((img_siz,img_siz),Image.LANCZOS).convert("RGB")
    if(flatten):
        if(norm):
            return (np.asarray(img).astype(np.float32).flatten()-127.5)/127.5
            
        else:
            return np.asarray(img).astype(np.float32).flatten()
         
    else:
        if(norm):
            return (np.asarray(img).astype(np.float32)-127.5)/127.5
        else:
            return np.asarray(img).astype(np.float32)
def crop_img(img,az=0.05):
    w,h=img.size
    ww=int(w*az)
    hh=int(h*az)
    ret=[]
    for x in [0,ww]:
        for y in [0,hh]:
            ret.append(img.crop((x,y,x+w-ww-1,y+h-hh-1)))
    return ret
def load_datas(rots=None,add_rand=None,add_crop=None,eps=0.1,sample=None,norm=False,**kwargs):
    from glob import glob
    from PIL import Image
    if(rots is None):
        rots=[0,90,180,270]
    poss=list(glob(path.join(curpth,'datas','pos','*')))
    negs=list(glob(path.join(curpth,'datas','neg','*')))
    xs=[]
    ys=[]
    #for i in random.sample(list(poss),70):
    allimg=[(i,[eps,1-eps]) for i in poss]+[(j,[1-eps,eps]) for j in negs]
    
    az=list(range(len(allimg)))
    #random.shuffle(az)
    allimg.sort(key=lambda x:base32(x[0],length=3))
    le=len(az)
    data=az[le//15:]
    test=az[:le//15]
    if(sample is not None):
        data=random.sample(data,sample)
    _xdata=[]
    _ydata=[]
    _xtest=[]
    _ytest=[]
    S=dict()
    siz=kwargs.get('img_siz',48)
    prog=0
    tm=time.time()
    ims=kwargs.get("img_siz",img_siz)
    tot=len(data)*len(rots)+len(test)
    for pth,y in [allimg[i] for i in data]:
        try:
            prog+=len(rots)
            print('load%.2f%%,remaining%.1fsec'%(prog*100/tot,(time.time()-tm)/prog*(tot-prog)),end='\r')
            
            im=None
            #h=myhash.phashi(im,w=40)
            #f=open(pth,'rb')
            h=path.basename(path.dirname(pth))+base32(pth,length=16)
            #f.close()
            if(h in S):
                print(pth,'==',S[h])
                try:
                    im.close()
                    os.remove(pth)
                    im=Image.open(S[h])
                except Exception as e:
                    print(e)
                    pass
                continue
            S[h]=pth
            for r in rots:
                
                #azz=[im1]
                
                svname="file=%s,rot=%d,size=%d.npy"%(h,r,ims)
                svpth=path.join(curpth,'npy',svname)
                if(path.exists(svpth)):
                    data=np.load(svpth)
                else:
                    if(im is None):
                        im=Image.open(pth)
                    im1=im.rotate(r,expand=True)
                    data=img2input(im1,**kwargs)
                    np.save(svpth,data)
                '''if(add_crop is None):
                else:
                    azz=[im1]+crop_img(im1,add_crop)'''
                
                _xdata.append(data)
                _ydata.append(y)
        except Exception as e:
            print(pth,e)
            #traceback.print_exc()
            continue
    #for i in random.sample(list(negs),70):
    for pth,y in [allimg[i] for i in test]:
        try:
            prog+=1
            print('load%.2f%%,remaining%.1fsec'%(prog*100/tot,(time.time()-tm)/prog*(tot-prog)),end='\r')
            h=path.basename(path.dirname(pth))+base32(pth,length=16)
            svname="file=%s,rot=%d,size=%d.npy"%(h,0,ims)
            svpth=path.join(curpth,'npy',svname)
            if(path.exists(svpth)):
                data=np.load(svpth)
            else:
                im=Image.open(pth)
                data=img2input(im,**kwargs)
                np.save(svpth,data)
            _xtest.append(data)
            _ytest.append(y)
        except Exception as e:
            print(pth,e)
    if(add_rand is not None):
        _xdata=np.array(_xdata,np.float32)
        az=_xdata+(np.random.rand(*_xdata.shape)-0.5)*add_rand
        _xdata=np.concatenate((_xdata,az),axis=0)
        _ydata=np.array(_ydata)
        _ydata=np.concatenate((_ydata,_ydata),axis=0)
    if(norm):
        _xdata=(np.array(_xdata,np.float16)-127.5)/127.5
        _xtest=(np.array(_xtest,np.float16)-127.5)/127.5
    if(not isinstance(_xdata,np.ndarray)):
        _xdata=np.array(_xdata,np.float16)
    return _xdata,np.array(_ydata),np.array(_xtest),np.array(_ytest)

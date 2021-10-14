#import tensorflow.keras as keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,InputLayer,Dropout,Conv2D,MaxPooling2D,Flatten
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,DepthwiseConv2D,BatchNormalization
from keras.layers import Concatenate,Activation,SeparableConv2D,Input,GlobalAveragePooling2D,Add
from keras.initializers import RandomNormal
import sys
sys.path.append(r'C:\YaqianBot\yaqianok\utils')
import myhash,myio
from os import path
import os
from process_data import load_datas
img_siz=128    #width=height=img_siz,RGB
n_input=img_siz*img_siz*3

n_classes=2    #sfw=0/nsfw=1

curpth=path.dirname(__file__)


initializer=RandomNormal(mean=0, stddev=5)

def add_tiny_conv(model,n,size,strides=(1,1)):
    w,h=size
    sw,sh=strides
    model.add(DepthwiseConv2D((1,h),strides=(1,sh),activation=None,use_bias=False,data_format='channels_last',kernel_initializer=initializer))
    model.add(DepthwiseConv2D((w,1),strides=(sw,1),activation=None,use_bias=False,data_format='channels_last',kernel_initializer=initializer))
    #model.add(BatchNormalization())
    #model.add(Activation("relu"))
    model.add(Conv2D(n,(1,1),activation='relu',data_format='channels_last'))
    return model
def add_tiny_conv_bn(model,n,size):
    w,h=size
    model.add(DepthwiseConv2D((1,h),activation=None,use_bias=False,data_format='channels_last'))
    model.add(DepthwiseConv2D((w,1),activation=None,use_bias=False,data_format='channels_last'))
    model.add(Conv2D(n,(1,1),activation=None,data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
def summary_and_exit():
    model.summary()
    from io import BytesIO
    f=BytesIO()
    keras.utils.plot_model(model,to_file='tmp.png',show_shapes=True)
    from PIL import Image
    im=Image.open('tmp.png')
    im.show()
    exit()
    
class cb(keras.callbacks.Callback):
    def __init__(self,*args,**kwargs):
        self.enmiao=None
        self.miao=None
        self.not_improving_cnt=0
        super().__init__(*args,**kwargs)
    def on_epoch_end(self, epoch, data):
        eps=1e-3
        acc=data['val_categorical_accuracy']
        model.save(path.join(savedir,'%.2f.h5'%(acc*100)))
        if(self.enmiao is None):
            self.enmiao=acc
        delta=acc-self.enmiao
        self.enmiao=acc
        if(self.miao is None):
            self.miao=eps*10
        self.miao=self.miao*0.95+delta*0.05
        if(self.miao<eps):
            print('not improving')
            self.not_improving_cnt+=1
        if(self.not_improving_cnt>10):
            print('stop')
            exit()
        print("miao=%.2f, delta=%.2f"%(self.miao/eps,delta/eps))
        
def add_m_inception(inp,n,m,use_tiny=True,merge='add'):
    outs=[]
    outs.append(Conv2D(n,(1,1),padding='same',data_format='channels_last',activation='relu')(inp))
    for i in range(1,m):
        out=inp
        for m in range(i):
            if(use_tiny):
                out=DepthwiseConv2D((1,3),padding='same',data_format='channels_last',activation=None,use_bias=False)(out)
                out=DepthwiseConv2D((3,1),padding='same',data_format='channels_last',activation=None,use_bias=False)(out)
                out=Conv2D(n,(1,1),padding='same',data_format='channels_last',activation='relu')(out)
            else:
                out=SeparableConv2D(n,(3,3),padding='same',data_format='channels_last',activation='relu')(out)
        outs.append(out)
    if(merge=='concat'):
        return Concatenate()(outs)
    else:
        return Add()(outs)
def add_inception(intensor,n):
    
    out0=Conv2D(n,(1,1),padding='same',data_format='channels_last',activation='relu')(intensor)
    
    out1=SeparableConv2D(n,(3,3),padding='same',data_format='channels_last',activation='relu')(intensor)
    
    tmp=SeparableConv2D(n,(3,3),padding='same',data_format='channels_last',activation='relu')(intensor)
    out2=SeparableConv2D(n,(3,3),padding='same',data_format='channels_last',activation='relu')(tmp)
    
    return Concatenate()([out0,out1,out2])
adagrad=keras.optimizers.Adagrad(lr=0.0055)
nadam=keras.optimizers.Nadam(lr=0.001)
if(True):
    #input=keras.Input((img_siz,img_siz,3))
    
    lr=0.0055
    input=Input((img_siz,img_siz,3))
    out=add_m_inception(input,4,3)  #7x7
    out=MaxPooling2D((2,2))(out)
    out=add_m_inception(out,8,3)  #5x5
    out=MaxPooling2D((2,2))(out)
    out=add_m_inception(out,16,3)  #5x5
    out=MaxPooling2D((2,2))(out)
    out=add_m_inception(out,32,3)  #5x5
    out=MaxPooling2D((2,2))(out)
    out=add_m_inception(out,64,3)  #5x5
    out=MaxPooling2D((2,2))(out)
    out=add_m_inception(out,128,2)  #3x3
    out=GlobalAveragePooling2D()(out)
    out=Dense(16,activation='relu')(out)
    out=Dense(2,activation='softmax')(out)
    model=keras.Model(input,out)
    #summary_and_exit()
    
    #summary_and_exit()
    #summary_and_exit()
    # opti=keras.optimizers.Adadelta(lr=lr,rho=0.75)
    opti=keras.optimizers.Adagrad(lr=0.0057)
    #opti=keras.optimizers.Nadam()
    
    
    #opti=keras.optimizers.SGD(lr=0.0057, momentum=0.9, nesterov=True)
    
    
    
    model.compile(loss='categorical_crossentropy',optimizer=opti,metrics=['categorical_accuracy'])
    ls=list()
    model.summary(print_fn=ls.append)
    summary='\n'.join(ls)
    name=myhash.hashs(summary)+"_lr=%.4f"%lr
    savedir=path.join(curpth,name)
else:
    #pth=r"M:\Weiyun Sync\code\nsfw_keras\5A2IYojGhykC4u1" #92.00%
    #pth=r"M:\Weiyun Sync\code\nsfw_keras\2bgA3muVIMk8cwK"
    #pth=r"M:\Weiyun Sync\code\nsfw_keras\23K5WpEg3ym2w6C"
    
    #savedir=pth=r"M:\Weiyun Sync\code\nsfw_keras\23K5WpEg3ym2w6C"
    savedir=pth=r'M:\Weiyun Sync\code\nsfw_keras\5AVYojP4jue2QwD_lr=0.0055'
    from glob import glob
    from os import path
    pth=list(glob(path.join(pth,'*.h5')))
    pth=sorted(pth)[-1]
    print(pth)
    model=load_model(pth)
    model.load_weights(pth)
    print(model.optimizer)
    #opti=keras.optimizers.Adagrad(lr=0.0055)
    '''model.summary()
    exit()'''
    #model.compile(loss='categorical_crossentropy',optimizer=opti,metrics=['categorical_accuracy'])
    ls=list()
    model.summary(print_fn=ls.append)
    summary='\n'.join(ls)
    #name=myhash.hashs(summary)+"_lr=%.4f"%lr
model.summary()

#exit()



print(savedir)
myio.savetext(path.join(savedir,'summary.txt'),summary)
keras.utils.plot_model(model,to_file=path.join(savedir,'plot.jpg'),show_shapes=True)
savepth=path.join(savedir,'nn.h5')
if(not path.exists(savedir)):
    os.makedirs(savedir)


#opti=keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
def shed():
    az=0
    def inner(epo):
        nonlocal az
        #from math import cos,pi
        import random
        if((az+1)%13 == 0):
            ret=-0.0007
        else:
            ret=random.random()*0.06
        if(0.99**az<0.07):
            az=0
        ret=ret*(0.99**az)
        
        print('lr=%.4f'%ret)
        az+=1
        
        return ret
    return inner
#cb=keras.callbacks.LearningRateScheduler(shed())

#opti=keras.optimizers.SGD(lr=0.004,momentum=0.02)

xtrain,ytrain,xtest,ytest=load_datas(rots=[0,90,180],flatten=False,img_siz=img_siz)
print(len(xtrain))
print(len(ytrain))
print(len(xtest))
print(len(ytest))

CB=cb()
for i in range(114514):
    if(i==1):
        #model.optimizer=nadam
        model.compile(loss='categorical_crossentropy',optimizer=nadam,metrics=['categorical_accuracy'])
    az=model.fit(xtrain,ytrain,batch_size=128,epochs=16,validation_data=(xtest,ytest),callbacks=[CB])
    print('fit',az)
    print('evaluate')
    loss,acc=model.evaluate(xtest,ytest,batch_size=64)
    print('epoch=%d,loss=%.2f,acc=%.2f%%'%(i,loss,acc*100))
    model.save(path.join(savedir,'%.2f.h5'%(acc*100)))
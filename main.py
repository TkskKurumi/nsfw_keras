#import tensorflow.keras as keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,InputLayer,Dropout,Conv2D,MaxPooling2D,Flatten
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras,time
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,DepthwiseConv2D,BatchNormalization
from keras.layers import Concatenate,Activation,SeparableConv2D,Input,GlobalAveragePooling2D,Add,Lambda





from keras.initializers import RandomNormal
import keras.backend as K
import sys
sys.path.append(r'C:\YaqianBot\yaqianok\utils')
import myhash,myio
from os import path
import os
from process_data import load_datas
img_siz=224    #width=height=img_siz,RGB
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
        eps=2e-4
        acc=data['val_categorical_accuracy']
        if(data['val_categorical_accuracy']>data['categorical_accuracy']):
            self.not_improving_cnt=max(0,self.not_improving_cnt-0.5)
        model.save(path.join(savedir,'%.2f.h5'%(acc*100)))
        if(self.enmiao is None):
            self.enmiao=acc
        delta=acc-self.enmiao
        self.enmiao=acc
        if(self.miao is None):
            self.miao=eps*5
        self.miao=self.miao*0.9+delta*0.1
        if(self.miao<eps):
            
            self.not_improving_cnt+=1
            print('not improving',self.not_improving_cnt)
        if(self.not_improving_cnt>8):
            print('stop')
            exit()
        else:
            self.not_improving_cnt=max(0,self.not_improving_cnt-0.5)
        print("miao=%.2f, delta=%.2f, not_improving_cnt=%d"%(self.miao/eps,delta/eps,self.not_improving_cnt))

def add_m_inception(inp,n,m,use_tiny=False,merge='concat',bn=False,short='nin'):
    outs=[]
    
    if(short=='nin'):
    
        if(bn):
            out=Conv2D(n,(1,1),padding='same',data_format='channels_last',activation='relu')(inp)
            out=BatchNormalization()(out)
            out=Activation('relu')(out)
            outs.append(out)
        else:
            outs.append(Conv2D(n,(1,1),padding='same',data_format='channels_last',activation='relu')(inp))
    elif(short=='mp'):
        out=MaxPooling2D((2,2),strides=(1,1))(inp)
        outs.append(out)
    elif(short is None):
        outs.append(inp)
    for i in range(1,m):
        out=inp
        for m in range(i):
            out=DepthwiseConv2D((3,3),padding='same',data_format='channels_last',activation=None,use_bias=False)(out)
            '''if(use_tiny):
                out=DepthwiseConv2D((1,3),padding='same',data_format='channels_last',activation=None,use_bias=False)(out)
                out=DepthwiseConv2D((3,1),padding='same',data_format='channels_last',activation=None,use_bias=False)(out)
                if(bn):
                    out=Conv2D(n,(1,1),padding='same',data_format='channels_last',activation=None)(out)
                    out=BatchNormalization()(out)
                    out=Activation('relu')(out)
                else:
                    out=Conv2D(n,(1,1),padding='same',data_format='channels_last',activation='relu')(out)
            else:
                if(bn):
                    out=SeparableConv2D(n,(3,3),padding='same',data_format='channels_last',activation=None,use_bias=False)(out)
                    out=Activation("relu")(BatchNormalization()(out))
                else:
                    out=SeparableConv2D(n,(3,3),padding='same',data_format='channels_last',activation='relu')(out)'''
        outs.append(Conv2D(n,(1,1),padding='same',data_format='channels_last',activation='relu')(out))
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
lr=0.002
adagrad=keras.optimizers.Adagrad(lr=lr*10)
nadam=keras.optimizers.Nadam(lr=lr)
continue_training=True
if(not continue_training):
    #input=keras.Input((img_siz,img_siz,3))
    
    input=Input((img_siz,img_siz,3))
    
    outs=[]
    #out=ImageNormalizeLayer()(input)
    out=add_m_inception(input,4,3)  #3x3 in 224x224
    outs.append(GlobalAveragePooling2D()(out))
    out=MaxPooling2D((2,2))(out)
    
    
    out=add_m_inception(out,8,3)  #3x3 in 112x112
    outs.append(GlobalAveragePooling2D()(out))
    out=MaxPooling2D((2,2))(out)
    
    
    out=add_m_inception(out,16,3)  #3x3 in 56x56
    outs.append(GlobalAveragePooling2D()(out))
    out=MaxPooling2D((2,2))(out)
    
    out=add_m_inception(out,32,3)  #3x3 in 28x28
    outs.append(GlobalAveragePooling2D()(out))
    
    out=MaxPooling2D((2,2))(out)
    
    out=add_m_inception(out,64,3)  #3x3 in 28x28
    outs.append(GlobalAveragePooling2D()(out))
    
    
    
    out=Concatenate()(outs)
    out=Dropout(0.5)(out)
    out=Dense(16,activation='relu')(out)
    
    out=Dropout(0.3)(out)
    out=Dense(2,activation='softmax')(out)
    model=keras.Model(input,out)
    #summary_and_exit()
    
    #summary_and_exit()
    #summary_and_exit()
    # opti=keras.optimizers.Adadelta(lr=lr,rho=0.75)
    #opti=keras.optimizers.Adagrad(lr=0.0057)
    #opti=keras.optimizers.Nadam()
    
    
    #opti=keras.optimizers.SGD(lr=0.0057, momentum=0.9, nesterov=True)
    
    
    
    model.compile(loss='categorical_crossentropy',optimizer=nadam,metrics=['categorical_accuracy'])
    ls=list()
    model.summary(print_fn=ls.append)
    summary='\n'.join(ls)
    name=myhash.hashs(summary)
    savedir=path.join(curpth,name)
else:
    savedir=pth=r'M:\Weiyun Sync\code\nsfw_keras\2jiSGv5GCMEOQqD'
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
from yield_data import yield_data,distribution
#opti=keras.optimizers.SGD(lr=0.004,momentum=0.02)
xval,yval=yield_data(mode='val',augment=None,smooth_label=None)
print('val',*distribution(yval))
next_yield=0
def yield_train():
    global next_yield,xtrain,ytrain
    if(time.time()<next_yield):
        return xtrain,ytrain
    tm=time.time()
    xtrain,ytrain=yield_data(mode='train',sample=len(yval)*4)
    tm=time.time()-tm
    next_yield=time.time()+tm*2
    dist=distribution(ytrain)
    print('train',*dist)
    return xtrain,ytrain
epochs_per_i=4
CB=cb()
for i in range(114514):
    xtrain,ytrain=yield_train()
    model.fit(xtrain,ytrain,batch_size=32,validation_data=(xval,yval),callbacks=[CB],initial_epoch=epochs_per_i*i,epochs=epochs_per_i*(i+1))
    
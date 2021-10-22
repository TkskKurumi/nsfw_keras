
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,DepthwiseConv2D,BatchNormalization
from keras.layers import Concatenate,Activation,SeparableConv2D,Input,GlobalAveragePooling2D,Add,Lambda
from keras.layers import AveragePooling2D


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
        outs.append(Conv2D(n,(1,1),padding='same',data_format='channels_last',activation='relu')(out))
    if(merge=='concat'):
        return Concatenate()(outs)
    else:
        return Add()(outs)
conv2d_common_args=(lambda **kwargs:kwargs)(padding='same',data_format='channels_last',activation='relu')
conv2d_common=lambda n,size:SeparableConv2D(n,size,**conv2d_common_args)
def dense_block(input,width,depth=3):
    outs=[]
    out=Conv2D(width,(1,1),**conv2d_common_args)(input)
    outs=[out]
    
    for i in range(depth):
        out=SeparableConv2D(width,(3,3),**conv2d_common_args)(out)
        outs.append(out)
        out=Concatenate()(outs)
    return out
def dense_net(img_siz=224):
    input=Input((img_siz,img_siz,3))
    outs=[]
    
    out=dense_block(input,8)                    #5x5 in 224x224
    outs.append(GlobalAveragePooling2D()(out))
    
    out=MaxPooling2D((2,2))(out)
    out=dense_block(out,8)                     #5x5 in 112x112
    outs.append(GlobalAveragePooling2D()(out))
    
    out=MaxPooling2D((2,2))(out)
    out=dense_block(out,16)                     #5x5 in 56x56
    outs.append(GlobalAveragePooling2D()(out))
    
    out=MaxPooling2D((2,2))(out)
    out=dense_block(out,32)                     #5x5 in 28x28
    outs.append(GlobalAveragePooling2D()(out))
    
    out=MaxPooling2D((2,2))(out)
    out=dense_block(out,64)                     #5x5 in 14x14
    outs.append(GlobalAveragePooling2D()(out))
    
    
    
    out=Concatenate()(outs)
    out=Dropout(0.5)(out)
    out=Dense(16,activation='relu')(out)
    
    out=Dropout(0.5)(out)
    out=Dense(2,activation='softmax')(out)
    
    return keras.Model(input,out)
    
def my_dense_net(img_siz=224,width=8):
    input=Input((img_siz,img_siz,3))
    out=input
    outs=[]
    
    
    out=conv2d_common(width,(3,3))(out)     #3x3 in 224x224
    out=conv2d_common(width,(3,3))(out)     #3x3 in 224x224
    out=conv2d_common(width,(3,3))(out)     #3x3 in 224x224
    outs.append(out)
    
    outs=[AveragePooling2D((2,2))(i) for i in outs]
    out=outs[0]
    out=conv2d_common(width,(3,3))(out)    #3x3 in 112x112
    out=conv2d_common(width,(3,3))(out)    #3x3 in 112x112
    out=conv2d_common(width,(3,3))(out)    #3x3 in 112x112
    outs.append(out)
    
    outs=[AveragePooling2D((2,2))(i) for i in outs]
    out=Concatenate()(outs)
    out=conv2d_common(width<<1,(3,3))(out)
    out=conv2d_common(width<<1,(3,3))(out)
    out=conv2d_common(width<<1,(3,3))(out)    #3x3 in 56x56
    outs.append(out)
    
    outs=[AveragePooling2D((2,2))(i) for i in outs]
    out=Concatenate()(outs)
    out=conv2d_common(width<<2,(3,3))(out)    #3x3 in 28x28
    out=conv2d_common(width<<2,(3,3))(out)    #3x3 in 28x28
    outs.append(out)
    
    outs=[AveragePooling2D((2,2))(i) for i in outs]
    out=Concatenate()(outs)
    out=conv2d_common(width<<3,(3,3))(out)    #3x3 in 14x14
    out=conv2d_common(width<<3,(3,3))(out)    #3x3 in 14x14
    outs.append(out)
    
    outs=[GlobalAveragePooling2D()(i) for i in outs]
    out=Concatenate()(outs)
    out=Dropout(0.2)(out)
    out=Dense(32,activation='relu')(out)
    
    out=Dropout(0.2)(out)
    out=Dense(2,activation='softmax')(out)
    
    return keras.Model(input,out)
    
def pretrained_mobilenet(img_siz=224):
    input=Input((img_siz,img_siz,3))
    base=keras.applications.MobileNet(include_top=False)
    base.trainable=False
    out=base(input)
    out=GlobalAveragePooling2D()(out)
    out=Dropout(0.3)(out)
    out=Dense(32,activation='relu')(out)
    out=Dropout(0.3)(out)
    out=Dense(2,activation='softmax')(out)
    model=keras.Model(input,out)
    return model
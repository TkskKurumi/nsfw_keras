#import keras
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,InputLayer,Dropout
from os import path
import os
from process_data import load_datas
img_siz=64	#width=height=img_siz,RGB
n_input=img_siz*img_siz*3
n_hidden0=100
n_hidden1=100
n_classes=2	#sfw=0/nsfw=1

curpth=path.dirname(__file__)
savedir=path.join(curpth,f'{n_input}_dense{n_hidden0}_dense{n_hidden1}')
savepth=path.join(savedir,'nn.h5')
if(not path.exists(savedir)):
    os.makedirs(savedir)

xtrain,ytrain,xtest,ytest=load_datas()
print(len(xtrain))
print(len(ytrain))
print(len(xtest))
print(len(ytest))
model=Sequential()
model.add(InputLayer(input_shape=(n_input,)))
#model.add(Dropout(0.07))
model.add(Dense(n_hidden0,activation='relu'))
model.add(Dense(n_hidden1,activation='relu'))
model.add(Dense(n_classes,activation='softmax'))
#from keras.optimizers import SGD
opti=keras.optimizers.Adagrad()
model.compile(loss='binary_crossentropy',optimizer=opti,metrics=['accuracy'])
for i in range(100):
    print('fit')
    model.fit(xtrain,ytrain,batch_size=64,epochs=20)
    print('evaluate')
    loss,acc=model.evaluate(xtest,ytest,batch_size=64)
    print('epoch=%d,loss=%.2f,acc=%.2f%%'%(i,loss,acc*100))
    if((i+1)%3==0):
        model.save(savepth)
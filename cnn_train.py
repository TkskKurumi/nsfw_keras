import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,InputLayer,Dropout,Conv2D,MaxPooling2D,Flatten
from os import path
import os
from process_data import load_datas
img_siz=64	#width=height=img_siz,RGB
n_input=img_siz*img_siz*3
n1,w1,h1=16,5,5
n2,w2,h2=12,5,5
n3,w3,h3=8,5,5
n_classes=2	#sfw=0/nsfw=1

curpth=path.dirname(__file__)
savedir=path.join(curpth,f'{n_input}_conv{n1},{w1},{h1}_conv{n2},{w2},{h2}_mp2,2_conv{n3},{w3},{h3}_mp2,2_adagrad0.006')
savepth=path.join(savedir,'nn.h5')
if(not path.exists(savedir)):
    os.makedirs(savedir)

xtrain,ytrain,xtest,ytest=load_datas(flatten=False,img_siz=img_siz)
print(len(xtrain))
print(len(ytrain))
print(len(xtest))
print(len(ytest))
model=Sequential()
model.add(InputLayer(input_shape=(img_siz, img_siz, 3)))
model.add(Dropout(0.07))
model.add(Conv2D(n1,(w1,h1),activation='relu',data_format='channels_last',input_shape=(img_siz, img_siz, 3)))
model.add(Conv2D(n2,(w2,h2),activation='relu'))
model.add(Dropout(0.07))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(n3,(w3,h3),activation='relu'))
model.add(Dropout(0.07))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(n_classes,activation='softmax'))
opti=keras.optimizers.Adagrad(lr=0.007)
model.compile(loss='categorical_crossentropy',optimizer=opti,metrics=['categorical_accuracy'])
for i in range(500):
    print('fit')
    model.fit(xtrain,ytrain,batch_size=32,epochs=20)
    print('evaluate')
    loss,acc=model.evaluate(xtest,ytest,batch_size=32)
    print('epoch=%d,loss=%.2f,acc=%.2f%%'%(i,loss,acc*100))
    model.save(path.join(savedir,'%.2f.h5'%(acc*100)))
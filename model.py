import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

data = np.load('data.npy')
labels = np.load('labels.npy')

model = Sequential()

model.add(Conv2D(200,(3,3),activation='relu',input_shape=(100,100,1)))
model.add(MaxPool2D(2,2))

model.add(Conv2D(100,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

checkpoint = ModelCheckpoint('mask.model',monitor='val_loss',verbose=0,save_best_only='True',mode='auto')

model.fit(data,labels,callbacks=[checkpoint],epochs=20,validation_split=0.2)


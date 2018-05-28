#Script for data acquisition and training the FCNN model
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time

#exec time 
start_time=time.time()


#reading train data from file 
data = pd.read_csv("optdigits.tra", header=None)
print(data.shape)

#reshaping Train data into a list of flattened arrays ands storing in df_x
df_x = data.iloc[:,0:64].values.reshape(len(data),64)
print(df_x.shape)


#Storing the Train labels in y
y = data.iloc[:,64].values
#print(y)


#converting Train labels into a one-of-c representation 
df_y = keras.utils.to_categorical(y,num_classes=10)



df_x = np.array(df_x)
df_y = np.array(df_y)


#print(df_x)
#print(df_y)
#df_x.shape


#Split test data and train data
x_train, x_val, y_train, y_val = train_test_split(df_x,df_y,test_size=0.2)
print(x_train.shape)

#FCNN model
#****MODEL BEGINS****

model = Sequential()

model.add(Dense(1000, activation='tanh', input_shape=[64]))
model.add(Dense(1000, activation='tanh'))
model.add(Dense(1000, activation='tanh'))
model.add(Dense(1000, activation='tanh'))


model.add(Dense(1000, activation='tanh'))
model.add(Dense(500, activation='tanh'))
model.add(Dense(500, activation='tanh'))
model.add(Dense(10, activation='softmax'))
#****END OF MODEL****
model.summary()

#setting model compilation parameters. Error function = crossentropy/sumofsquares
keras.optimizers.SGD(lr=0.001, momentum=0.001, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


#training model
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=50)

#Early Stopping condition to avoid overfitting
keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0,mode='auto')

print("--- Total training time %s (s) ---" % (time.time() - start_time))	
#training complete: save trained parameters to file 
model.save_weights('FCNNWeights.h5')

#Script for predicting test data with CNN Model 
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
import sys

def network_model():
    #****MODEL BEGINS****
    model = Sequential()

    model = Sequential()
    model.add(Convolution2D(32, 3, data_format='channels_last', activation='relu', input_shape=(8,8,1)))
    model.add(MaxPooling2D(2, 2))

    #model.add(Convolution2D(32, 2, activation='relu'))
    #model.add(MaxPooling2D(1, 1))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    print("CNN Model Summary:\n")
    model.summary()
    #****END OF MODEL****
    print("\n\n")
    return model

#reading test file 
testData = pd.read_csv("optdigits.tes", header=None)

#reshaping Test data into 8x8 arrays ands storing in df_x_t
df_x_t = testData.iloc[:,0:64].values.reshape(len(testData),8,8,1)

#Storing the Test labels in y_t
y_t = testData.iloc[:,64].values

#converting Test labels into a one-of-c representation 
df_y_t = keras.utils.to_categorical(y_t,num_classes=10)

df_x_t = np.array(df_x_t)
df_y_t = np.array(df_y_t)

#print(df_x_t.shape)
#print(df_y_t.shape)

model = network_model()
model.load_weights('./CNNWeights.h5')

#reading train data from train file 
data = pd.read_csv("optdigits.tra", header=None)
#reshaping Train data into 8X8 arrays ands storing in df_x
df_x = data.iloc[:,0:64].values.reshape(len(data),8,8,1)

#Storing the Train labels in y
y = data.iloc[:,64].values
#converting Train labels into a one-of-c representation 
df_y = keras.utils.to_categorical(y,num_classes=10)

df_x = np.array(df_x)
df_y = np.array(df_y)

print("******TRAIN DATA PERFORMANCE*****")
prediction = model.predict(df_x)
prediction_label=np.zeros(3823)

#Decoding softmax output
for x in range(0,3823):
    predictClass=0
    confidence=-1
    for i in range(0,10):
        if(prediction[x,i]>confidence):
            predictClass=i
            confidence=prediction[x,i]
    prediction_label[x]=predictClass;

#print(prediction[0])

#print(prediction_label[722])
#print(y_t[722])

#calculating confusion matrix
confusionMatrix = np.zeros((10,10))
for x in range(0,3823):
    m=int(y[x])
    n=int(prediction_label[x])
    confusionMatrix[m,n]=int(confusionMatrix[m,n])+1


sum=0
for x in range(0,10):
    sum+=confusionMatrix[x,x]
overallAccuracy=(sum/3823)*100
S="Overall Accuracy obtained from CNN is: " +repr(overallAccuracy)+"%"
print(S)
print("\n")
classAccuracy=np.zeros(10)
for x in range(0,10):
    sum=0
    TP=int(confusionMatrix[x,x])
    for y in range(0,10):
        sum+=int(confusionMatrix[x,y])
    classAccuracy[x]=(TP/sum)*100
    S="Class Accuracy for digit '"+repr(x)+ "' is: " +repr(classAccuracy[x])+"%"
    print(S)
print("\n")
S="Confusion Matrix for CNN classification: \n"
print(S)
print(confusionMatrix)

print("\n\n\n")

print("******TEST DATA PERFORMANCE*****")
prediction = model.predict(df_x_t)
prediction_label=np.zeros(1797)

#decoding softmax output
for x in range(0,1797):
    predictClass=0
    confidence=-1
    for i in range(0,10):
        if(prediction[x,i]>confidence):
            predictClass=i
            confidence=prediction[x,i]
    prediction_label[x]=predictClass;

#print(prediction[0])

#print(prediction_label[722])
#print(y_t[722])

#calculating confusion matrix
confusionMatrix = np.zeros((10,10))
for x in range(0,1797):
    m=int(y_t[x])
    n=int(prediction_label[x])
    confusionMatrix[m,n]=int(confusionMatrix[m,n])+1


sum=0
for x in range(0,10):
    sum+=confusionMatrix[x,x]
overallAccuracy=(sum/1797)*100
S="Overall Accuracy obtained from CNN is: " +repr(overallAccuracy)+"%"
print(S)
print("\n")
classAccuracy=np.zeros(10)
for x in range(0,10):
    sum=0
    TP=int(confusionMatrix[x,x])
    for y in range(0,10):
        sum+=int(confusionMatrix[x,y])
    classAccuracy[x]=(TP/sum)*100
    S="Class Accuracy for digit '"+repr(x)+ "' is: " +repr(classAccuracy[x])+"%"
    print(S)
print("\n")
S="Confusion Matrix for CNN classification: \n"
print(S)
print(confusionMatrix)
import os
curdir=os.getcwd()

import numpy as np
import random 
import keras
from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Flatten,Dense,Dropout,BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 


seed=7
np.random.seed(seed)


def data_loader(filename,size,seed):
    
    data_loc=curdir+"/data/"+filename
    dataset=np.loadtxt(data_loc,delimiter=',')
    np.random.shuffle(dataset)
    total_data=dataset.shape[0]
    train=int(.75*total_data)
    
    
    dataX=dataset[:,:-1].reshape((dataset.shape[0],size,size,1))
    dataY=dataset[:,-1].reshape((dataset.shape[0],1))
    
    dataX=dataX/255.0
    dataX, X_test, Y_train, Y_test = train_test_split(dataX,dataY,test_size=0.25,random_state=seed)
    return dataX, X_test, Y_train, Y_test

def create_model(size):
    
    model=Sequential()
    
    model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=((size,size,1))))
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2,2)))
    #model.add(MaxPooling2D(pool_size=((2,2))))
    model.add(Dropout(0.25))

    
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2,2)))
    #model.add(MaxPooling2D(pool_size=((2,2))))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2,2)))
    #model.add(MaxPooling2D(pool_size=((2,2))))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    
    return model
    
def diff_measurements(A):
    
    accuracy=A.trace()/A.sum()
    
    precision=np.zeros((1,A.shape[1]))
    recall=np.zeros((1,A.shape[1]))
    
    for i in range(A.shape[0]):
        precision[0,i]=A[i,i]/sum(A[:,i])
        recall[0,i]=A[i,i]/sum(A[i])
        
    return accuracy,precision,recall
    

# Load the data & split them into test & train set

filename='data_3_class.csv'
size=64
dataX, X_test, Y_train, Y_test=data_loader(filename,size,seed)

Y_test=to_categorical(Y_test)
Y_train=to_categorical(Y_train)

# Train the model or load the model

retrain=False # to retrain make retrain= True 
batch_size=25
epochs=50

if (retrain==True):
    model=create_model(size)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    history=model.fit(dataX,Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1)
    
    saved_file=curdir+"/model/"+"xNet_3_class.hdf5" #While retraining change the name 
    model.save(saved_file)
    
    plt.plot(history.history['accuracy'],label="Test Accuracy")
    plt.plot(history.history['val_accuracy'],label="Validation Accuracy")
    plt.legend()

else:
    saved_model=curdir+"/model/"+"xNet_3_class.hdf5"
    model=load_model(saved_model)
    model.summary()
    

# Test the accuracy and other metrics

test_scores=model.predict(X_test,verbose=1)
matrix = confusion_matrix( test_scores.argmax(axis=1),Y_test.argmax(axis=1))
accuracy,precision,recall=diff_measurements(matrix)
print('Accuracy : ',accuracy)
print('Precission: ',precision)
print('Recall : ',recall)
plt.figure()
sns.heatmap(matrix,annot=True,fmt='d',cmap='Blues_r')
plt.title('Confusion Matrix for Test data')
plt.xlabel("True Label")
plt.ylabel("Prediceted Label")

# Choose 12 photos randomly from test set for visualization

random_num=random.sample(range(0,X_test.shape[0]),12)
pred_label=test_scores.argmax(axis=1)
actual_label=Y_test.argmax(axis=1)
plt.figure()
i=1
for number in random_num:
    img=(X_test[number].reshape((size,size)))*255.0
    Plabel=pred_label[number]
    Alabel=actual_label[number]
    plt.subplot(3,4,i)
    plt.subplots_adjust(hspace=0.4)
    plt.imshow(img,cmap="gray")
    plt.subplot(3,4,i).set_title("True Label : "+str(Alabel)+" Prediceted Label : "+str(Plabel))
    i=i+1
    
    

plt.show()

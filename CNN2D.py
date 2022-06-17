import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Dropout, Flatten,Activation,Dense
from keras import optimizers

#preprocessing
X = pd.read_excel('/content/drive/MyDrive/Luận_văn_PTDL/Train/Output_Final.xlsx',sheet_name ='data')
sc = StandardScaler()
X_scaled  = sc.fit_transform(X)
Y =  pd.read_excel('/content/drive/MyDrive/Luận_văn_PTDL/Train/Output_Final.xlsx',sheet_name ='Label')
Y_label = LabelEncoder().fit_transform(Y)
y_data = np_utils.to_categorical(Y_label)

X_data = X_scaled.reshape(2100,322,8,1)
X_test = X_test.reshape(1,322,8,1)

#trainning
batch_size = 8
model = Sequential()
model.add(Conv2D(16,(3,3), padding = 'same'))
model.add(MaxPooling2D(3,3))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(AveragePooling2D(2,2))
model.add(Flatten())

model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_data, y_data, batch_size=batch_size, epochs=10,validation_split = 0.1)
#evaluate
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(50)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy of CNN 2D')

plt.subplot(2, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss Of CNN 2D')
plt.show()

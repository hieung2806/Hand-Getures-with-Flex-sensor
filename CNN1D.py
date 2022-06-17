import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout,Conv1D, MaxPooling1D,GlobalAveragePooling1D, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
import umap.umap_ as umap
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D

#preprocessing
X = pd.read_excel('/content/drive/MyDrive/Luận_văn_PTDL/Train/Output_Final.xlsx', sheet_name = 'data')
Y = pd.read_excel('/content/drive/MyDrive/Luận_văn_PTDL/Train/Output_Final.xlsx', sheet_name = 'Label')
X_test = pd.read_excel('/content/drive/MyDrive/Luận_văn_PTDL/Train/test_data_final_2.xlsx', sheet_name ='data')
Y_test = pd.read_excel('/content/drive/MyDrive/Luận_văn_PTDL/Train/test_data_final_2.xlsx', sheet_name ='Label' )
X_test = StandardScaler().fit_transform(X_test)
Y_test = LabelEncoder().fit_transform(Y_test)
ss = StandardScaler()
X = ss.fit_transform(X)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)
Y_test_encoder = np_utils.to_categorical(Y_test)
img_rows, img_cols = 322,8
X_data = X.reshape(2100, img_rows, img_cols)
X_test = X_test.reshape(1400, img_rows, img_cols)
input_shape = (img_rows, img_cols)

#train model
BATCH_SIZE = 8
model_m = Sequential()
model_m.add(Conv1D(8, 6, activation='relu', input_shape= input_shape))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(16, 6, activation='relu'))
model_m.add(GlobalAveragePooling1D(name='G_A_P_1D'))
model_m.add(Flatten())
model_m.add(Dense(32))
model_m.add(Activation('relu'))
model_m.add(Dropout(0.5))
model_m.add(Dense(7, activation='softmax'))
model_m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model_m.fit(X_data, dummy_y, batch_size=BATCH_SIZE, epochs=50, validation_split=0.1)
#evaluate model
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
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#test_model
y_pred = model_m.predict(X_test)
model_m.evaluate(X_test,Y_test_encoder)
label = ['Vâng','Dừng','Đưa','Uống','Ăn','None','Không']
cm = confusion_matrix(Y_test,y_pred.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
disp.plot(cmap=plt.cm.Blues)
plt.show()
print(precision_recall_fscore_support(Y_test, y_pred.argmax(axis=1), average='macro'))


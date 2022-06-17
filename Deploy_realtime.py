import serial
import time
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

gestures_name = {0:'Uong',1:'An',2:'Dua',3:'Khong',4:'None',5:'Dung',6:'Vang'}
sc = StandardScaler()

model = load_model('CNN_2D.h5')

def processing_data(X):
    X_data = sc.fit_transform(X)
    return X_data

def predict(X):
    result = model.predict(X_data)
    return result
#
ser = serial.Serial('COM6', 9600, timeout=1)
ser.close()
ser = serial.Serial('COM6', 9600, timeout=1)
time.sleep(2)

data = np.empty([0,8])

for i in range(322): # test doc serial 2 lan
    # make sure the 'COM#' is set according the Windows Device Manager   
    line = ser.readline()   # read a byte string  
    data_length = 8 # 8 values = 3 acc + 5 flex
    #      
    string_tmp = line.decode()  # convert the byte string to a unicode string    
    string_tmp = string_tmp.strip() # remove '\r\n'
    line_Data = []
    for i in range(data_length-1):
        first_semicolon = string_tmp.find(';') # find the position of the first occurrence of ';'
        extractedString =  string_tmp[0:first_semicolon] # cat lay chuoi string dau tien 
        #
        strTmp = ''
        for tmp in extractedString: 
            strTmp = strTmp + tmp
        instant_value = float(strTmp)
        line_Data = np.append(line_Data, instant_value)
        # remove extracted string and ';'
        string_tmp = string_tmp.replace(extractedString+';','')  
    line_Data = np.append(line_Data, float(string_tmp))    
    #    
    # print(line_Data)
    line_Data = np.reshape(line_Data, (1,8))
    data = np.append(data, line_Data, axis = 0)    
ser.close()
time.sleep(2)

X_data = processing_data(data)
X_data = np.reshape(data, (1,322,8,1))
result = model.predict(X_data)
print(gestures_name[np.argmax(result)])

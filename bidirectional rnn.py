#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
import random
from sklearn.metrics import mean_squared_error
import time
import pandas as pd
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import csv
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras.layers.recurrent import SimpleRNN, LSTM, GRU

import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

# In[2]:


df=pd.read_excel(r'C:\Users\aliakbar.DESKTOP-UBRBJ9M\Desktop\python\train.xls')


totalInput=df.drop('target',axis=1)
totalLabel=df['target']

totalInput = np.array(totalInput)
totalLabel = np.array(totalLabel)
min_max_scaler = MinMaxScaler()
totalInput = min_max_scaler.fit_transform(totalInput)
totalInput = np.reshape(totalInput, (totalInput.shape[0], 1, totalInput.shape[1]))


# In[3]:


import matplotlib.pyplot as plt
model = Sequential()
model = Sequential()
model.add(Bidirectional(GRU(8)))
model.add(Dense(36, activation='linear'))
model.add(Dense(12, activation='linear'))
model.add(Dense(1, activation='linear'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history =model.fit(totalInput, totalLabel,epochs=2000, batch_size =15, shuffle=True, validation_split=0.1)

# In[4]:


hist = history.history


# In[ ]:


y_pred = model.predict(totalInput)
mse = mean_squared_error(totalLabel, y_pred)
rmse = math.sqrt(mse)
print("Root Mean Square Error Total : {:.4f}".format(rmse))


# In[5]:

hist['epoch'] = history.epoch
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error')
plt.plot(hist['epoch'], hist['mean_absolute_error'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
         label = 'Val Error')
plt.ylim([0,5])
plt.legend()
plt.savefig('Mean Abs Error.png')
 
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
plt.plot(hist['epoch'], hist['mean_squared_error'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mean_squared_error'],
         label = 'Val Error')
plt.ylim([0,20])
plt.legend()
 
plt.show()




# In[ ]:
df2=pd.read_excel(r'C:\Users\aliakbar.DESKTOP-UBRBJ9M\Desktop\python\test.xls')


totalInput2=df2.drop('target',axis=1)
totalLabel2=df2['target']

totalInput2 = np.array(totalInput2)
totalLabel2 = np.array(totalLabel2)
min_max_scaler = MinMaxScaler()
totalInput2 = min_max_scaler.fit_transform(totalInput2)
totalInput2 = np.reshape(totalInput2, (totalInput2.shape[0], 1, totalInput2.shape[1]))



predict_Label2 = model.predict(totalInput2)


all=pd.read_excel(r'C:\Users\aliakbar.DESKTOP-UBRBJ9M\Desktop\python\all_point.xlsx')
all_InputData = np.array(all)
min_max_scaler = MinMaxScaler()
all_InputData = min_max_scaler.fit_transform(all_InputData)
all_InputData = np.reshape(all_InputData, (all_InputData.shape[0], 1, all_InputData.shape[1]))

predict_all = model.predict(all_InputData)

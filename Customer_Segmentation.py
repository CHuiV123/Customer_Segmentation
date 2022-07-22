#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:20:05 2022

@author: angela
"""

#%% IMPORTS 


from CS_module import GRAPH,Cramers,ModelDevelopment,ModelEvaluation
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler 
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential, Input 
from tensorflow.keras.utils import plot_model
from sklearn.impute import KNNImputer

import matplotlib.pyplot as plt 
import missingno as msno
import pandas as pd
import numpy as np 
import datetime
import pickle
import os 

#%% CONSTANTS 

CSV_PATH = os.path.join(os.getcwd(),'dataset','Train.csv')

LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))

MMS_PATH = os.path.join(os.getcwd(),'models','mms.pkl')

MODEL_SAVE_PATH = os.path.join(os.getcwd(),'models','model.h5')

#%% 1) Data Loading 

df = pd.read_csv(CSV_PATH)

#%% 2) Data Inspection 

df.info() # Total entries 31647 
df.describe().T
df.isna().sum()
df.duplicated().sum()
msno.matrix(df)
msno.bar(df)

df.columns


# To drop ID column since id column is not very useful in data analysis here. 
# To drop days_since_prev_campaign_contact has NaNs value >80%
df = df.drop(['id','days_since_prev_campaign_contact'], axis=1)

plt.figure(figsize=(15,12))
df.boxplot() # balance and last contact duration have outliers. outliers in balance should not be removed in this case as balance in bank is very much subjective. 

con = ['balance']

cat = list(df.drop(con, axis=1))

g = GRAPH()
g.plot_cat_graph(cat,df)
g.plot_con_graph(con,df)


#%% 3) Data Cleaning 

df.isna().sum()

le = LabelEncoder()
for i in cat: 
    if i == 'term_deposit_subscribed': 
        continue
    else: 
           le = LabelEncoder()
           temp = df[i]
           temp[temp.notnull()]=le.fit_transform(temp[df[i].notnull()])
           df[i]= pd.to_numeric(df[i], errors = 'coerce')
           PICKLE_SAVE_PATH = os.path.join(os.getcwd(),'pkl_model',i+'encoder.pkl')
           with open(PICKLE_SAVE_PATH, 'wb') as file: 
               pickle.dump(le,file)
i + '.pkl'


knn_im = KNNImputer()
df_imputed = knn_im.fit_transform(df)
df_imputed = pd.DataFrame(df_imputed)
df_imputed.columns = df.columns 
df_imputed['customer_age'] = np.floor(df_imputed['customer_age'])
df_imputed['personal_loan'] = np.floor(df_imputed['personal_loan'])
df_imputed['last_contact_duration'] = np.floor(df_imputed['last_contact_duration'])

df = df_imputed

df.info()
df.describe().T 
df.isna().sum()


#%% Feature Selection 


# cat vs cat 
#cramer's V 

# Target: term_deposit_subscribed
c = Cramers()

for i in cat: #categorical vs categorical 
    print(i)
    confusion_matrix_notfunc = pd.crosstab(df[i],df['term_deposit_subscribed']).to_numpy()
    print(c.cramers_corrected_stat(confusion_matrix_notfunc))


for i in con: #continuous vs categorical 
    print(i)
    lr=LogisticRegression()
    lr.fit(np.expand_dims(df[i],axis=-1),df['term_deposit_subscribed'])
    print(lr.score(np.expand_dims(df[i],axis=-1),df['term_deposit_subscribed']))


X = df.drop(labels=['term_deposit_subscribed',
                    'marital','education','default','housing_loan',
                    'personal_loan','communication_type','day_of_month',
                    'num_contacts_in_campaign'], axis =1)
# dropping term_deposit_subscribed because that supoose to be our target. 
# dropping 'marital','num_contacts_in_campaign','default','education' as they have very low correlational ratio

y = df['term_deposit_subscribed']


#%% 5) Data preprocessing 

mms = MinMaxScaler()
X = mms.fit_transform(X)
with open (MMS_PATH,'wb') as file: 
    pickle.dump(mms,file)

# one hot encoder for target
ohe = OneHotEncoder(sparse = False)
y = ohe.fit_transform(np.array(y).reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                   random_state= 123)

#%% Model Development 

nb_class = len(np.unique(y_train, axis = 0))
input_shape= np.shape(X_train)[1:]

md = ModelDevelopment()
model = md.simple_dl_model(input_shape,nb_class)


#%% Model Compilation 


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics = ['acc'])

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

early_callback = EarlyStopping(monitor = 'val_loss', patience=5) 

plot_model(model,show_shapes=(True))
 
#%% Model training 

hist = model.fit(X_train, y_train, epochs=100,
                 callbacks=[tensorboard_callback,early_callback],
                 validation_data = (X_test,y_test))

#%% model evaluation 

print(hist.history.keys())

me = ModelEvaluation()
me.LOSS_plot(hist)
me.ACC_plot(hist)

print(model.evaluate(X_test,y_test)) #model score

#%% Model Analysis 

pred_y = model.predict(X_test)

pred_y = np.argmax(pred_y, axis=1)
true_y = np.argmax(y_test, axis=1)

print(confusion_matrix(true_y, pred_y))

cm = confusion_matrix(true_y, pred_y)
cr = classification_report(true_y, pred_y) 

labels = ['Unsubscribe','Subscribe']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(cr)

# This dataset is an imbalance dataset, accuracy score may not indicate that 
# this model can do a very good prediction.  


#%% Model saving 

model.save(MODEL_SAVE_PATH)











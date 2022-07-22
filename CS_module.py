#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 10:13:52 2022

@author: angela
"""

#%% 

from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras import Sequential, Input
import matplotlib.pyplot as plt 
import scipy.stats as ss
import seaborn as sns 
import numpy as np

#%% 
class GRAPH(): 
    def plot_cat_graph(self,cat,df):
        '''
        ..... this function is meant to plot categorical data using seaborn function

        Parameters
        ----------
        cat : category list
            DESCRIPTION.
            df : dataframe
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        for i in cat: 
            plt.figure()
            sns.countplot(df[i])  # for categorical use count plot 
            plt.show()


    def plot_con_graph(self,con,df): 
        '''
        ..... this function is meant to plot continuous data using seaborn function 

        Parameters
        ----------
        con : continuous list
            DESCRIPTION.
            df : dataframe
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        for i in con: 
            plt.figure()
            sns.distplot(df[i])  # for continuous use distribution plot 
            plt.show()


class Cramers():
    def cramers_corrected_stat(self,confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher,
            Journal of the Korean Statistical Society 42 (2013): 323-328
            """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%%
class ModelDevelopment:
    def simple_dl_model(self, input_shape,nb_class,nb_node=64,dropout_rate=0.3):
        '''
        

        Parameters
        ----------
        input_shape : TYPE
            DESCRIPTION: input shape is the data shape of X train
        nb_class : TYPE
            DESCRIPTION: nb_class is the total of class we have in this dataset
        nb_node : TYPE, optional
            DESCRIPTION. The default is 64.
        dropout_rate : TYPE, optional
            DESCRIPTION. The default is 0.3.
        activation : TYPE, optional
            DESCRIPTION. 

        Returns
        -------
        None.

        '''
        model = Sequential()
        model.add(Input(shape=(input_shape)))
        model.add(Dense(nb_node, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_node, activation ='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class, activation= 'softmax'))
        model.summary()

        return model


class ModelEvaluation():
    def LOSS_plot(self,hist):
        plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.show()
    
    def ACC_plot(self,hist): 
        plt.figure()
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.show()

    
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:54:37 2021

@author: LENOVO
"""

import os
import pandas as pd  #manipulation data
import numpy as np #series
import matplotlib.pyplot as plt #visualisation
import seaborn as sns #visualisation

#####################################################################
###                    Set working directory                      ###
#####################################################################
# os module in python provides functions for interacting with the operating system
# Python method chdir() changes the current working directory to the given path.It returns None in all the cases.
os.chdir("C:\\Users\\LENOVO\\Desktop\\GitHub\\Project_1")
#####################################################################
###                        Fonctions utiles                       ###
#####################################################################

#Creation de nouveau classeur (repertoire)
def CreateDirectory(filename):
    try:
        os.mkdir(filename)
    except OSError:
        print ("! Directory already exist !")
    else:
        print ("Successfully created") 
    
#Fonction pour verifier la qualité de données 'creation de tableau Dataquality'
def dataquality(data,filename):
    data_quality = pd.DataFrame(columns=['Data Type','Non Missing Values','Unique Values','Missing Values',
                                         'Missing Percent','Mean','Min','Max','Q1','Q3','EIQ'])
    data_quality['Data Type']=data.dtypes
    data_quality['Non Missing Values']=data.shape[0]-data.isnull().sum()
    data_quality['Missing Values']=data.isnull().sum()
    data_quality['Missing Percentage']=round((data.isnull().sum()/data.shape[0])*100,2)
    data_quality['Unique Values'] = data.nunique()
    
    data_quality.Min, data_quality.Q1, data_quality.Mean, data_quality.Q3, data_quality.Max = round((data.quantile([0, 0.25, 0.5, 0.75, 1])),2).values.tolist()
    data_quality.EIQ= round(data_quality.Q3-data_quality.Q1 ,2)  
    data_quality.to_csv('Data\Processed\dataquality_'+ filename+ '.csv',index=True, sep = ';')        
        
        
        
#################################################################
###                  Exploratory Data Analysis                ###
#################################################################

data_train_original = pd.read_csv(r"Data\Original\test.csv", delimiter = ',')
data_train_brut = data_train_original.copy()
data_train_brut.head()
data_train_brut.tail()
data_train_brut.describe()
data_train_brut.shape
data_train_brut.columns
data_train_brut.nunique()  # numbre of unique values
data_train_brut["Parch"].unique()
data_train_brut.isnull().sum()

# relationship analysis 
corelation = data_train_brut.corr()
sns.heatmap(corelation, xticklabels = corelation.columns, yticklabels= corelation.columns, annot = True ) 
# Dendrogram 
# Using the denrdogram to find the optimal number of clustering 
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from pylab import rcParams

# from sklearn.preprocessing import LabelEncoder for categorical features
from sklearn.preprocessing import LabelEncoder
#
# Instantiate LabelEncoder
#
le = LabelEncoder()
#
# Encode single column status
#
data_train_brut.Name = le.fit_transform(data_train_brut.Name)
data_train_brut.Sex = le.fit_transform(data_train_brut.Sex)
data_train_brut.columns
data_train_brut.drop("Cabin",axis = 1,  inplace = True)
data_train_brut.drop("Ticket",axis = 1,  inplace = True)
# Print df.head for checking the transformation

data_train_brut.head
X = data_train_brut.dropna()
X = data_train_brut.iloc[:,:-1]
linked = linkage(X, 'single')
plt.rcParams.update({'font.size':9})
plt.figure(figsize=(10, 8))
dendrogram(linked,
            orientation='top',
            labels=list(X.columns),
            #distance_sort='descending',
            show_leaf_counts=True)
plt.show()
# to visualize relationship between two variables where variables can be continuous categorical
sns.pairplot(data_train_brut)
# scatter plot is a type of data display that shows the relationship between two numerical variables each member of data cell gets plotted as a point whose left parenthesis ''' named relationplot' 
sns.relplot(x= 'math_score', y = 'reading_score', hue= 'gender', data= data_train_brut)

# histogram is a display of data using powers of different heights and in a histogram each bar groups into ranges so the taller bars
# show that mode rate of range actually falls in that and a histogram basically displaced shape and the spread and continuous sample data 
# to undersand distributions and the boxplot for doing the same 
sns.distplot(data_train_brut[""])
# to plot a boxplot 
sns.catplot(x = ' ' , kind= 'box',  data = data_train_brut ) 











# -*- coding: utf-8 -*-


import os
import pandas as pd  #manipulation data
import numpy as np #series
import matplotlib.pyplot as plt #visualisation
import seaborn as sns #visualisation

 
#################################################################
###                  Exploratory Data Analysis                ###
#################################################################
def dataquality0(data,filename):
    data_quality = pd.DataFrame(columns=['Data Type','Non Missing Values','Unique Values','Missing Values',
                                         'Missing Percentage','Mean','Min','Max'])
    data_quality['Data Type']=data.dtypes
    data_quality['Non Missing Values']=data.shape[0]-data.isnull().sum()
    data_quality['Missing Values']=data.isnull().sum()
    data_quality['Missing Percentage']=round((data.isnull().sum()/data.shape[0])*100,2)
    data_quality['Unique Values'] = data.nunique()
    data_quality['Mean'] = data.mean()
    data_quality['Min'] = data.min()
    data_quality['Max'] = data.max()
    data_quality.to_csv('Data\Processed\dataquality_'+ filename+ '.csv',index=True, sep = ';') 
data_test_original = pd.read_csv(r"Data\Original\test.csv", delimiter = ',')
data_train_original = pd.read_csv(r"Data\Original\train.csv", delimiter = ',')
data_train_brut = data_train_original.copy()
data_test_brut = data_test_original.copy()
#check dataquality brut for data_train & data_test
dataquality0(data_train_brut,"brut_train")
dataquality0(data_test_brut,"brut_test")

#data_train_brut.head() data_train_brut.tail() data_train_brut.describe()
# Number of rows shape[0] and columns shape[1]
data_train_brut.shape
data_train_brut.nunique()  # numbre of unique values
data_train_brut["Parch"].unique()
data_train_brut.isnull().sum()
# checking out missing data!
data_train_brut.isnull()
sns.heatmap(data_train_brut.isnull(), yticklabels = False,cbar = False,  cmap = 'viridis')

# to visualize relationship between two variables where variables can be continuous categorical
#sns.pairplot(data_train_brut)
# scatter plot is a type of data display that shows the relationship between two numerical variables each member of data cell gets plotted as a point whose left parenthesis ''' named relationplot' 
sns.relplot(x= 'Age', y = 'Survived', hue= 'Sex', data= data_train_brut)

# Number of non _'survivers' by sexe
sns.set_style("darkgrid") # whitegrid
sns.countplot(x= 'Survived', hue = 'Sex', data= data_train_brut) 
# Number of non _'survivers' by Pclass
sns.set_style("darkgrid") # whitegrid
sns.countplot(x= 'Survived', hue = 'Pclass', data= data_train_brut, palette ='viridis' ) 
sns.countplot(x= 'SibSp', data= data_train_brut)
# histogram is a display of data using powers of different heights and in a histogram each bar groups into ranges so the taller bars
# show that mode rate of range actually falls in that and a histogram basically displaced shape and the spread and continuous sample data 
# to undersand distributions and the boxplot for doing the same 
sns.distplot(data_train_brut["Age"].dropna())
# to plot a boxplot 
sns.catplot(x = 'Age' , kind= 'box',  data = data_train_brut ) 
# boxplot for all variables 
ax = sns.boxplot(data=data_train_brut, orient="h", palette="Set2")
ax1 = sns.boxplot(data=data_train_brut, orient="h", palette="Set2")


#############################################
#            Data preprocessing             #
#############################################
# Handling Outliers 











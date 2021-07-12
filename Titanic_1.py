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
# get the count of the number of Survived
data_train_brut['Survived'].value_counts()
# Visualize the count of survivals
sns.countplot(data_train_brut['Survived'])
(data_train_brut.loc[data_train_brut['Sex'] == 'male']['Survived']).value_counts()
(data_train_brut.loc[data_train_brut['Pclass'] == 1]['Survived']).value_counts()
(data_train_brut.loc[(data_train_brut['Pclass'] == 1) & (data_train_brut['Sex'] == 'female')]['Survived']).value_counts()
#data_train_brut.head() data_train_brut.tail() data_train_brut.describe()
# Number of rows shape[0] and columns shape[1]
data_train_brut.shape
data_train_brut.nunique()  # numbre of unique values
data_train_brut["Parch"].unique()
data_train_brut.isnull().sum() # # Count empty values in each column # data_train_brut.isna().sum()

# checking out missing data!
data_train_brut.isnull()
data_train_brut.columns
sns.heatmap(data_train_brut.isnull(), yticklabels = False,cbar = False,  cmap = 'viridis')
# Siblings and spouse ,,,, parch = number of parents or children that are on the ship with the passenger and embarked just says where the person embarked from ' embarquer = monter à bord '
#visualize the count of survivors for columns 'who', sex, pclass,sibsp, parch, embarked
# let's create  a variable called cols which will be shortof columns and a list that contains the column names
cols = ['Sex','SibSp', 'Parch', 'Embarked']  # 2 rows and 3 columns to make 6 charts because we have 6 variables
n_rows = 2
n_cols = 2
# we will create now a subplot grid  and the figure size of each graph
fig, axs = plt.subplots(n_rows,n_cols, figsize = (n_cols * 3.2, n_rows * 3.2))
# look through each column
for r in range(0, n_rows):
    for c in range(0, n_cols):
        i = r*n_cols + c # index to go through the number of columns
        ax = axs[r][c] # show where to position each subplot
        sns.countplot(data_train_brut[cols[i]], hue = data_train_brut['Survived'], ax = ax)
        ax.set_title(cols[i])
        ax.legend(title = 'Survived', loc = 'upper right')
plt.tight_layout()  #  automatically adjust subplot params so the subplots fits in to the figure area
# look at survival rate'taux' by sex
data_train_brut.groupby('Sex')[['Survived']].mean()
#look at survival rate by sex and pclass
data_train_brut.pivot_table('Survived', index = 'Sex', columns = 'Pclass')
#look at survival rate by sex and pclass visually 
data_train_brut.pivot_table('Survived', index = 'Sex', columns = 'Pclass').plot() # what we saw above we can see it basically here using the graph
# plot each survival rate  of each pclass
sns.barplot(x= 'Pclass', y= 'Survived', data = data_train_brut )
# look at survival rate by sex, age, and class
age = pd.cut(data_train_brut['Age'], [0,18,80])
data_train_brut.pivot_table('Survived',['Sex', age],'Pclass')
# Plot the prices paid of each class by sex
sns.relplot(x= 'Fare', y = 'Pclass', hue = 'Sex', data= data_train_brut)
# Plot the prices paid of each class
plt.scatter(data_train_brut['Fare'], data_train_brut['Pclass'], color = 'Purple', label = 'Passenger Paid')
plt.ylabel('Pclass')
plt.xlabel('Price/ Fare')
plt.title('Price of each class')
plt.legend()
plt.show()
# look at all of the values in each column and get a count 
for val in data_train_brut:
    print(data_train_brut[val].value_counts())
    print()
    
data_train_brut.drop(['PassengerId','Name','Ticket', 'Cabin'], axis = 1, inplace = True)
data_train_brut.drop(['Cabin'], axis = 1, inplace = True)
# remove the rows  wwith missing values and columns in dataset'jeu de données'
data_train_brut.dropna(subset = ['Embarked','Age'], inplace = True)
data_train_brut.shape
# examine type of data
data_train_brut.dtypes

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











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
# examine type of data
data_train_brut.dtypes
# Now after checking the type of our data we can use a method in sklearn preprocession called labelencoder
# which transforms our object data to integers
from sklearn.preprocessing import LabelEncoder
# we will create a variable called 'labelencoder' equal to LabelEncoder method or constructor
labelencoder = LabelEncoder()
# Encode the sex column
data_train_brut.iloc[:,2] = labelencoder.fit_transform(data_train_brut.iloc[:,2])
# Encode the Embarked column
data_train_brut.iloc[:,7] = labelencoder.fit_transform(data_train_brut.iloc[:,7])
# Split the data into independent  'X' and dependent 'Y' variables
X = data_train_brut.iloc[:,1:8].values
y = data_train_brut.iloc[:,0].values 
# split the data into 80% training and 20% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
# Scale the data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
#######################################################
##     Function with many machinelearning models     ##
#######################################################
# Create  a function with many machine learning models
def models(X_train, y_train): 
    # Use Logistic Regression 
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, y_train)
    # Use Kneighbors
    from sklearn.neighbors import KNeighborsClassifier
    knn= KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p= 2)
    knn.fit(X_train, y_train)
    # Use SVC
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel = 'linear', random_state = 0)
    svc_rbf.fit(X_train, y_train) # for training 
    # Use GaussianNB
    from sklearn.naive_bayes import GaussianNB
    gauss =  GaussianNB()
    gauss.fit(X_train, y_train)
    # Use Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, y_train)
    # Use RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10,criterion = 'entropy', random_state = 0)
    forest.fit(X_train, y_train)
    # print the training accuracy for each model
    print('[0] Logistic Regression Training Accuracy:', log.score(X_train, y_train))
    print('[0] KNeighborsClassifier Training Accuracy:', knn.score(X_train, y_train))
    print('[0] SVC Training Accuracy:', svc_rbf.score(X_train, y_train))
    print('[0] GaussianNB Training Accuracy:', gauss.score(X_train, y_train))
    print('[0] DecisionTreeClassifier Training Accuracy:', tree.score(X_train, y_train))
    print('[0] RandomForestClassifier Training Accuracy:', forest.score(X_train, y_train))
    return log, knn, svc_rbf, gauss, tree, forest
# get and train all the models 
model = models(X_train, y_train)
# Show the confusion matrix and accuracy for all of the models on the test data 
from sklearn.metrics import confusion_matrix
for i in range(len(model)):
    cm = confusion_matrix(y_test, model[i].predict(X_test))
    # Extract TN ' true negative, false positif, FP, FN, TP 
    TN, FP, FN, TP = cm.ravel()
    test_score = (TP + TN) / (TP + TN + FN + FP)
    print(cm)
    print('Model[{}] Testing Accuracy = "{}"'.format(i, test_score))
    print()










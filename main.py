import pandas as panda
import matplotlib.pyplot as plot
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

train = panda.read_csv("train.csv") #Use pandas to read the comma seperated values

total = train.isnull().sum().sort_values(ascending=False) #Create sums for each count of missing data in each column
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False) #Create a percentage from that
missing_data = panda.concat([total, percent], axis=1, keys=['Total', 'Percent']) #Creates a new table that contains the sorted list of all columns with no data and how much is missing
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1) #Removes all columns with more than 1 bit of missing data
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max() #Check that there is no more

#Applying log transformations to fixed skewness.
train['SalePrice'] = np.log(train['SalePrice'])
train['GrLivArea'] = np.log(train['GrLivArea'])
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index) #Create a new column with as many rows as TotalBsmtSF
train['HasBsmt'] = 0 #By default all rows have the value of 0 (False, they have no basement)
train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1 #UNLESS, the totalbsmtsf is more than 0, implying there is a basement, so change to 1 (True)
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF']) #Wherever there IS a basement, apply the log transformation to the TotalBsmtSF col

#convert categorical columns to dummies
train = panda.get_dummies(train)
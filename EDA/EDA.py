import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('datasets.csv', encoding_errors='ignore')
data.shape
data.tail(2)
data.shape
data.info()
data.describe()
data.isnull().sum()
#dropping all missing value rows
data.isnull().sum()
data.duplicated().sum()
data[data.duplicated()]
data.drop_duplicates(inplace=True)
data.duplicated().sum()
data[data.duplicated()]

data.dtypes
data['id']= data['id'].astype(object)


data['host_id']= data['host_id'].astype(object)


'''**univarate analysis**'''
#Price
data['price']
sns.histplot(data=data, x='price')
sns.boxplot(data=data, x='price')
df = data[data['price']<1500]

sns.boxplot(data=df, x='price')
sns.histplot(data=df, x='price')

plt.figure(figsize=(8,5))
sns.histplot(data=df, x='price',bins=100)
plt.title("Price Distribution")
plt.ylabel("Frequency")
plt.show()

df.columns
df.dtypes

plt.figure(figsize=(6,3))
sns.histplot(data=df, x='availability_365')
plt.title("availability_365 Distribution")
plt.ylabel("Frequency")
plt.show()

df.groupby(by='neighbourhood_group')['price per bed'].mean()

df.groupby(by='neighbourhood_group')['price'].mean()

#feature engineering
#['price per bed']
df['price per bed'] = df['price'] / df['beds']
df.head(2)


#bivariate analysis
df.columns
sns.barplot(data=df,x='neighbourhood_group', y='price', hue='room_type')

#no of reviews and price relationship
plt.figure(figsize=(8,5))
plt.title("locality and review dependency")
sns.scatterplot(data=df,x='number_of_reviews',y='price', hue='neighbourhood_group')
plt.show()


df.dtypes

sns.pairplot(data=df, vars=['price','minimum_nights','number_of_reviews','availability_365'], hue='room_type')

sns.scatterplot()
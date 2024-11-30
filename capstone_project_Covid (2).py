#!/usr/bin/env python
# coding: utf-8

# 

# In[2]:


# importing required libraries
import pandas as pd
import numpy as np
import matplotlib_inline as plt
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#importing the dataset 
covid = pd.read_csv("covid_19_clean_complete.csv")


# In[4]:


# displaying the information on the dataset 
covid


# In[5]:


#using describe  method to check about the dataset 
covid.describe()


# In[6]:


#checking count values of dataset using count method
covid.count()


# In[7]:


# using value_counts method to check number of occurance   
covid.value_counts()


# In[8]:


# checking for number of rows and columns in the dataset using shape method
covid.shape


# In[9]:


# checking for category of columns in the dataset
covid.columns


# In[10]:


# checking for dublicated information in the dataset 
covid.duplicated().sum()


# In[11]:


# checking for empty values in the dataset
covid.isna().sum()


# In[12]:


# dropping irrelevant column in the dataset
covid.drop(["Province/State"],axis=1,inplace= True)


# In[13]:


# reviewing the dataset if columns has been dropped
covid


# In[14]:


# renaming Country/Region column to Country for easy assesing 
covid.rename(columns=  {"Country/Region":"Country"},inplace=True) 


# In[15]:


#converting date column to datetime stamp for extraction
covid["Date"] = pd.to_datetime(covid["Date"])


# In[16]:


# extracting Month from datetime in numerics for modelling using dt.month
covid["Month"] = covid["Date"].dt.month


# In[17]:


# extracting Month from datetime in string for visualization using dt.month_name()
covid["Months"] = covid["Date"].dt.month_name()


# In[18]:


# extracting day from datetime in numerics for modelling using dt.day
covid["Day"] = covid["Date"].dt.day


# In[19]:


# extracting day from datetime in strings for visualization using dt.day_name
covid["Days"] = covid["Date"].dt.day_name()


# In[20]:


# displaying the dataset to check if new features are included
covid


# In[21]:


# Merging longitude and latitude columns into a single column
covid['Coordinates'] = covid.apply(lambda row: (row['Lat'], row['Long']), axis=1)


# In[22]:


# sorting date in ascending order 
covid["Date"] = covid["Date"].sort_values()

#setting date to index 
covid.set_index("Date",inplace=True)


# In[23]:


# sorting country names alphabetically 
covid.sort_values(by="Country",ascending=True,inplace=True)


# In[24]:


#sorting countries based on date index
covid.sort_values(by=["Country","Date"],ascending=True,inplace=True)


# In[25]:


# displaying dataset to see new added features 
covid


# In[26]:


# extracting_daily_growth rate in percentage from the given information in dataset
covid["Daily growth rate"] = ((covid["Confirmed"] - covid["Confirmed"].shift(1))/(covid["Confirmed"].shift(1))*100)


# In[27]:


# displaying information to see new features 
covid


# In[28]:


# extracting Total_population from the given information in the dataset 
covid["Total_Population"] = (covid["Confirmed"]+covid["Deaths"]+covid["Recovered"]+covid["Active"])


# In[29]:


# extracting Mortality_rate from the given information in the dataset
covid["Mortality rate"] = (covid["Deaths"] / covid["Total_Population"])*100 


# In[30]:


# displaying the dataset to see the new features 
covid


# In[31]:


covid["Daily Recovery"] = (covid["Recovered"] - covid["Recovered"].shift(1))/(covid["Recovered"].shift(1))*100


# In[32]:


covid.tail()


# In[33]:


covid["Daily Active"] = (covid["Active"] - covid["Active"].shift(1))/(covid["Active"].shift(1))*100


# In[34]:


covid.tail()


# In[35]:


# grouping confirmed cases per country population using groupby method
case_per_population = covid.groupby("Country")["Confirmed"].sum().reset_index()


# In[36]:


# displaying grouped cases base on country 
case_per_population


# In[37]:


# grouping countries base on their Regions 
WHO_Regions = covid.groupby("Country")["WHO Region"].sum().reset_index()


# In[38]:


# displaying grouped countries base on their regions 
WHO_Regions


# In[39]:


# checking for missing values in the dataset 
covid.isna().sum()


# In[40]:


# filling missing values with 0 instead of NAN 
covid.fillna({"Daily growth rate":0},inplace=True)

covid["Daily growth rate"]


# In[41]:


# checking if missing values are filled
covid.isna().sum()


# In[42]:


# filling the missing values with 0  
covid.fillna({"Mortality rate": (int(0))},inplace=True)
covid["Mortality rate"]


# In[43]:


#checcking dataset if missing values has been filled 
covid.isna().sum()


# In[44]:


# filling the missing values with 0
covid.fillna({"Daily Recovery": (int(0))},inplace=True)
covid.fillna({"Daily Active":   (int(0))},inplace=True) 


# In[45]:


covid.isna().sum()


# In[46]:


covid.duplicated().sum()


#importing labelEncoder for convertint string values to numerics
from sklearn.preprocessing import LabelEncoder


# In[88]:


#creating a new column with converted country and Region to numerics
encorder = LabelEncoder()
covid["EncCountry"] = encorder.fit_transform(covid["Country"]) 
covid["EncWHO Region"] = encorder.fit_transform(covid["WHO Region"]) 


# In[90]:


# dropping duplicates 
covid.drop_duplicates(inplace=True)


# In[91]:


#checking for duplicated values 
covid.duplicated().sum()


# In[93]:


# replace all infinty values with non-numerics and drop all non numerics 
#covid = covid.replace([np.inf, -np.inf], np.nan)dropna()
covid = covid.replace([np.inf, -np.inf], np.nan).fillna(int(0))


# In[95]:


#spliting the dataset into train and test 
x = covid[["Long","Lat","Confirmed","Deaths","Recovered","Active",
           "Daily Active","Daily Recovery","Total_Population","Mortality rate","Day","Month","EncWHO Region","EncCountry"]]
y = round(covid["Daily growth rate"])


# In[98]:


# checking for the shape of the dataset, number of columns and rows
covid.shape


# In[99]:


# converting y and x values to descreate values kusing round function to round up figures
x = np.array((x.round()))
y = np.array((y.round()))


# In[102]:


# displaying dataset to see featues 
covid


# In[103]:


# importing necessary libraries for assigning splitied values into features and target column for machine learning model 
import sklearn      
from sklearn.model_selection import train_test_split


# In[106]:


# assigning splited dataset to test and train
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[135]:


# importing classification model for model building 
#from sklearn.ensemble import RandomForestCl

from sklearn.tree import DecisionTreeClassifier


# In[137]:


#using estimator to improve model performance assigning and assigning to a variable 
#model = RandomForestClassifier(n_estimators=100, max_depth=10)#n_estimators=100, max_depth=10

model = DecisionTreeClassifier() 


# In[139]:


# fitting model 
print(model.fit(x_train,y_train))


# In[140]:


# predicting model performance 
y_pred = model.predict(x_test).astype(int)
print(y_pred)


# In[141]:


#evaluating model accuracy
from sklearn.metrics import accuracy_score


# In[142]:


# printing model accuracy 
accuracy = accuracy_score(y_test,y_pred)
print(f"Model Accuracy: {accuracy:.2f}%")


# In[143]:


from sklearn.metrics import classification_report, accuracy_score, precision_score,confusion_matrix,f1_score,recall_score


# In[144]:


# Confusion Matrix
report = confusion_matrix(y_test, y_pred)
print(report)


# In[145]:


# Print classification report
print(classification_report(y_test,y_pred))


# In[146]:


#printing model precsion score
precision = precision_score(y_test,y_pred,average="weighted")
print(f"precision score : {precision:.4f}")


# In[147]:


# printing model F1 score"
F1_score = f1_score(y_test,y_pred,average="weighted")
print(f"F1 Score :{F1_score : 2f}")


# In[148]:


#printing model recall score
recall = recall_score(y_test,y_pred, average="micro")
print(f"recall : {recall:.2f} ") 


# In[ ]:





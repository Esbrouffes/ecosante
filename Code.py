# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 14:03:36 2021

@author: YBlachonpro
"""


def takeYear(str):
    return str[-4:]
def takeCountry(str):
    c=str.find(';')
    
    return str[:c]
def convert(str):
    return float(str)
def normalize(number):
    if type(number)==float or type(number)==int:
        return number * 100/ max_val
def root(number):
    return number**(1/2)

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

HLY= pd.read_csv('data.csv').drop([0,1],axis=0)

HLY.rename(columns={'Unnamed: 0':'country','Healthy life expectancy (HALE) at birth (years).1': "HLY0", 'Healthy life expectancy (HALE) at age 60 (years).1':'HLY60'}, inplace = True)
HLY=HLY[['country','HLY0','HLY60']]##année = 2015
HLY["HLY0"]=HLY["HLY0"].apply(convert)
HLY["HLY60"]=HLY["HLY60"].apply(convert)

BDR = pd.read_csv("BDR.csv")

BDR["Year"]=BDR['Country; Year'].apply(takeYear)
BDR["Country"]=BDR['Country; Year'].apply(takeCountry)
BDR.rename(columns={'Infant mortality rate (probability of dying between birth and age 1 per 1000 live births); Both sexes':'IMR1','Under-five mortality rate (probability of dying by age 5 per 1000 live births); Both sexes':'IMR5'},inplace=True)
BDR=BDR[BDR['Year']=='2015'][['Country', 'Year',"IMR1","IMR5"]]


df=HLY.merge(BDR,left_on='country', right_on='Country').fillna(0)
df.drop('country',axis=1,inplace=True)


SCI=pd.read_csv("SCI.csv")
SCI.rename(columns={'UHC index of service coverage (SCI).1':"SCI",'Unnamed: 0':'Country'},inplace=True)
SCI=SCI[["Country","SCI"]]
### => Low/medium availability => faible proportion

df=df.merge(SCI,left_on='Country', right_on='Country').fillna(0)

CHE=pd.read_csv("CHE.csv")
#CHE.drop(0)
CHE.rename(columns={'Unnamed: 0':'Country','Current health expenditure (CHE) as percentage of gross domestic product (GDP) (%).3':"CHE"},inplace=True)
CHE=CHE[["CHE","Country"]]
CHE.drop([0])

df=df.merge(CHE,left_on="Country",right_on="Country")


Med=pd.read_csv("Med.csv")
Med=pd.read_csv("Med_2.csv")

for i in [2015,2014,2016]:
        
    countries=Med[Med["Year"]==i]["Country"]
    
    for c in countries:
        for index,row in Med.iterrows():
            if row["Country"] ==c and row["Year"]!=i:
                Med.drop(index,inplace=True)
Med.drop("Year",axis=1,inplace=True)
Med.rename(columns={'Skilled health professionals density (per 10 000 population)':"Skilled"},inplace=True)
for index, row in Med.iterrows():
        if row["Skilled"]=="No data":
            Med.drop(index,inplace=True)
Med["Skilled"]=Med["Skilled"].apply(convert)    
df=df.merge(Med,left_on="Country",right_on="Country")


for col in df.columns:
    if col not in ["Year","Country"]:
            max_val=df[col].max()
            df[col]=df[col].apply(normalize)
    
    
df["indice_composite"]=0.20*(100-df["HLY0"])+0.10*(100-df["HLY60"])+0.2*(100-df["IMR5"])+0.1*(100-df['IMR1'])+0.2*df["SCI"]+0.2*df["Skilled"]
#print(df[df["indice_composite"]>=60]["Country"])
df["root"]=df["CHE"].apply(root)

df.dropna(inplace=True)
X=np.array(df["CHE"])#,"root"]]
#X_poly=sklearn.preprocessing.PolynomialFeatures(3).fit_transform(X)
"""
X_train,X_test=sklearn.model_selection.train_test_split(X,test_size=0.01)
y_train,y_test=sklearn.model_selection.train_test_split(df["indice_composite"],test_size=0.01)
model=sklearn.linear_model.LinearRegression().fit(X_train,y_train)

plt.figure()

plt.scatter(df["CHE"],model.predict(X))"""



import scipy as sp
from scipy.optimize import curve_fit

def func(x,a,b):
    return a*(x-b)**(1/2)
popt, pcov = sp.optimize.curve_fit(func, X, df["indice_composite"])
popt
plt.scatter(df["CHE"],df["indice_composite"])
plt.plot(X, func(X, *popt), 'r-')

df.to_csv("DataGrouped.csv")
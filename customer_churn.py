# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:54:49 2020

@author: Melih Görgülü
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# read csv file
df = pd.read_csv("telco_churn.csv")

# 7043X21 DataFrame

# lets explore data

df.sample(10)

df.columns
# columns == ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
#           'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
#           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
#           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
#           'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

describe_all_df = df.describe(include="all").T

df.describe().T

def find_standard_error(mean,std):
    return std / (mean**(1/2))

# tenure : Number of months the customer has stayed with the company
# tenure mean,std: 32.371149,24.559481

# montly charges mean,std:64.761692, 30.090047


print("\nTenure standard error: " + str(find_standard_error(df["tenure"].mean(),df["tenure"].std())))

print("\nMontly Caherges standard error: " + str(find_standard_error(df["MonthlyCharges"].mean(),df["MonthlyCharges"].std())))

for i in df.columns:
    print(i + " dtype: " + str(df[i].dtype))
    
    
# DATA MANIPULATION

df.isna().sum() # there are not NA values in the data

# there are some empty values in total charges, lets fix it

for index,value in enumerate(df["TotalCharges"]):
    if(value==" "):
        df.drop(index,axis=0,inplace=True) # drop empty records

df["TotalCharges"] = df["TotalCharges"].astype("float32") # convert from string to float


# replace 'No internet service' to No for the following columns

replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
for i in replace_cols : 
    df[i]  = df[i].replace({'No internet service' : 'No'})


#replace values
df["SeniorCitizen"] = df["SeniorCitizen"].replace({1:"Yes",0:"No"})

# LETS VİSUALİZE DATA

# what percenge of customers churned or not 
fig,ax  = plt.subplots()
plt.pie(df["Churn"].value_counts(),labels=["No","Yes"],autopct='%1.1f%%')
plt.title("Percentage of Churned or Not",fontdict={"weight":"bold"})

# mean tenure of churned or not churned customers

churned_tenure = df.groupby("Churn").mean()["tenure"]
churned_tenure.name = "Mean Tenure"

ax = churned_tenure.plot(kind="barh",legend=True,colormap="RdPu_r",fontsize=14)

plt.title("Mean Tenure by Churned or Not",fontdict={"fontsize": 14,"weight":"bold"})
plt.ylabel("Churn",fontdict={"fontsize": 14,"weight":"bold"})

# mean total charges of churned or not churned customers
churned_total_charge = df.groupby("Churn").mean()["TotalCharges"]
churned_total_charge.name = "Mean Total Charge"

churned_total_charge.index = ["No({0})".format(int(churned_total_charge["No"])),"Yes({0})".format(int(churned_total_charge["Yes"]))]

ax = churned_total_charge.plot(kind="barh",legend=True,colormap="viridis_r",fontsize=14)

plt.title("Mean Total Charge by Churned or Not",fontdict={"fontsize": 14,"weight":"bold"})
plt.ylabel("Churn",fontdict={"fontsize": 14,"weight":"bold"})

# gender distribution in customer attrition

fig ,ax = plt.subplots(1,2)
churned_gender = df[df["Churn"]=="Yes"].groupby("gender").count()["Churn"]

not_churned_gender = df[df["Churn"]=="No"].groupby("gender").count()["Churn"]

ax[0].pie(churned_gender,labels=["Female","Male"],autopct='%1.1f%%')
ax[0].set_title("Churned")
ax[1].pie(not_churned_gender,labels=["Female","Male"],autopct='%1.1f%%')
ax[1].legend()

ax[1].set_title("Not Churned")

# senior distribution in customer attrition
fig ,ax = plt.subplots(1,2)
fig.suptitle('Churned and not churned user by Senior citizen feature', fontsize=16)
churned_senior = df[df["Churn"]=="Yes"].groupby("SeniorCitizen").count()["Churn"]

not_churned_senior = df[df["Churn"]=="No"].groupby("SeniorCitizen").count()["Churn"]

ax[0].pie(churned_senior,labels=["Not Senior Citizen","Senior Citizen"],autopct='%1.1f%%')
ax[0].set_title("Churned")
ax[1].pie(not_churned_senior,labels=["Not Senior Citizen","Senior Citizen"],autopct='%1.1f%%')
ax[1].legend()

ax[1].set_title("Not Churned")

# Internet Service dist in customer attrition
fig ,ax = plt.subplots(1,2)
fig.suptitle('Internet Service dist in customer attrition', fontsize=16)
churned_internet = df[df["Churn"]=="Yes"].groupby("InternetService").count()["Churn"]

not_churned_internet = df[df["Churn"]=="No"].groupby("InternetService").count()["Churn"]

ax[0].pie(churned_internet,labels=["DSL","Fiber","No"],autopct='%1.1f%%')
ax[0].set_title("Churned")
ax[1].pie(not_churned_internet,labels=["DSL","Fiber","No"],autopct='%1.1f%%')
ax[1].legend()

ax[1].set_title("Not Churned")

# contract distribution in customer attrition
fig ,ax = plt.subplots(1,2)
fig.suptitle('Contract dist in customer attrition', fontsize=16)
churned_contract = df[df["Churn"]=="Yes"].groupby("Contract").count()["Churn"]

not_churned_contract = df[df["Churn"]=="No"].groupby("Contract").count()["Churn"]

ax[0].pie(churned_contract,labels=["Month-to-Month","One Year","Two Year"],autopct='%1.1f%%')
ax[0].set_title("Churned")
ax[1].pie(not_churned_contract,labels=["Month-to-Month","One Year","Two Year"],autopct='%1.1f%%')
ax[1].legend()

ax[1].set_title("Not Churned")

# payment method distribution in customer attrition
fig ,ax = plt.subplots(1,2)
fig.suptitle('Payment Method dist in customer attrition', fontsize=16)
churned_payment = df[df["Churn"]=="Yes"].groupby("PaymentMethod").count()["Churn"]

not_churned_payment = df[df["Churn"]=="No"].groupby("PaymentMethod").count()["Churn"]

ax[0].pie(churned_payment,labels=["Bank transfer","Credid Card","Electronic Check","Mailed Check"],autopct='%1.1f%%')
ax[0].set_title("Churned")
ax[1].pie(not_churned_payment,labels=["Bank transfer","Credid Card","Electronic Check","Mailed Check"],autopct='%1.1f%%')
ax[1].legend()

ax[1].set_title("Not Churned")

# lets create tenure_interval for better understanding

def set_tenure_interval(x):
    if x <= 12 :
        return "0-12"
    elif (x > 12 and x <= 24 ):
        return "12-24"
    elif (x > 24 and x <= 48) :
        return "24-48"
    elif (x > 48 and x <= 60) :
        return "48-60"
    elif x > 60 :
        return "60+"

df["tenure_interval"] = df["tenure"].apply(lambda x:set_tenure_interval(x))

# tenure month interval distribution in customer attrition
fig ,ax = plt.subplots(1,2)
fig.suptitle('Tenure mont interval dist in customer attrition', fontsize=16)

churned_tenure_interval = df[df["Churn"]=="Yes"].groupby("tenure_interval").count()["Churn"]
not_churned_tenure_interval = df[df["Churn"]=="No"].groupby("tenure_interval").count()["Churn"]

ax[0].pie(churned_tenure_interval,labels=["0-12 Month","12-24 Month","24-48 Month","48-60 Month","60+ Month"],autopct='%1.1f%%')
ax[0].set_title("Churned")
ax[1].pie(not_churned_tenure_interval,labels=["0-12 Month","12-24 Month","24-48 Month","48-60 Month","60+ Month"],autopct='%1.1f%%')
ax[1].legend()

ax[1].set_title("Not Churned")


# customer attrition based on tenure interval

churn_interval = df[df["Churn"]=="Yes"].groupby("tenure_interval").count()["Churn"]
not_churn_interval = df[df["Churn"]=="No"].groupby("tenure_interval").count()["Churn"]
not_churn_interval.name = "Non Churn"

churn_by_interval = pd.concat([churn_interval,not_churn_interval],axis=1)

ax = churn_by_interval.plot(kind="bar",color=["red","green"])

ax.set_title("Customer attrition based on tenure interval")
ax.set_xlabel("Tenure Interval(Month)")
ax.set_ylabel("Count")

# lets see gender and other numeric features relationship
ax = sns.pairplot(df[["tenure","MonthlyCharges","TotalCharges","gender"]],hue="gender")

# cant see big difference between numeric features by gender

# lets plot pairplot for see relationships between numeric variables
ax = sns.pairplot(df[["tenure","MonthlyCharges","TotalCharges","Churn"]],hue="Churn")

# let look at the corralation matrix for numeric variables

ax = sns.heatmap(df.corr(),linewidths=.5,annot=True,cmap="YlGnBu")


# DATA PREPROCESSİNG




# OUTLIER Analysis for numeric variables
# BOXPLOT

# tenure boxplot

sns.boxplot(df["tenure"],orient="v")

# churn and tenure

sns.boxplot(x="Churn",y="tenure",data=df)

# MontlyCharges boxplot
sns.boxplot(df["MonthlyCharges"],orient="v")

# churn and MontlyCharges

sns.boxplot(x="Churn",y="MonthlyCharges",data=df)

# TotalCharges boxplot
sns.boxplot(df["TotalCharges"],orient="v")
# Churn and TotalCharges

sns.boxplot(x="Churn",y="TotalCharges",data=df)


# LETS USE IQR RANGE METHOD ON TOTALCHARGES AND TENURE TO DETECT OUTLİERS


def find_outliers(series):
    q75 , q25 = np.percentile(series,[75,25])
    IQR = q75-q25
    lower_bound = q25-IQR*1.5
    upper_bound = q75 + IQR*1.5    
    
    outliers = {}
    for index,val in enumerate(series):
        if(val<lower_bound or val>upper_bound):
            outliers[index] = val
    
    outliers = pd.DataFrame(data= outliers.items(),columns=["Index",series.name])

    if(outliers.size==0):
        print("Can't find outlier value in that column")
    else:
        print("\nLower Bound:{0}\n-\n-\n-\n-\n-\nUpper Bound: {1}".format(lower_bound,upper_bound))
        return outliers
    
def delete_outliers(outliers_df):
    for idx in outliers_df["Index"]:
        df.drop(idx,inplace=True)
        print(str(idx)+ " deleted")


# CHURN-TENURE IQR RANGE ANALYSİS
        
tenure_churn_outliers = find_outliers(df[df["Churn"]=="Yes"]["tenure"])

# churn-monthlycharges outlier analysis
find_outliers(df[df["Churn"]=="Yes"]["MonthlyCharges"])
find_outliers(df[df["Churn"]=="No"]["MonthlyCharges"])


# there are some outliers values on TotaCharges

# lets only delete entitys whose cause outlier value on both TotalCharges and tenure columns

totalcharges_churn_outliers = find_outliers(df[df["Churn"]=="Yes"]["TotalCharges"])

outliers = totalcharges_churn_outliers.merge(tenure_churn_outliers,on = "Index" )

delete_outliers(outliers)


# ENCODE CATEGORIC VARIABLES

from sklearn.preprocessing import LabelEncoder


binary_variables = []
for col in df:
    if(len(df[col].value_counts()) == 2 and df[col].dtype=="O" and col !="customerID"):
        binary_variables.append(col)

# lets fit label encoder to the binary attributes
        
LE = LabelEncoder()
for col_name in binary_variables:
    df[col_name] = LE.fit_transform(df[col_name])
    print(col_name+ " is encoded")
    
    
# lets take categoric columns (not binary)

categoric_variables = []

for col in df:
    if(len(df[col].value_counts()) > 2 and df[col].dtype=="O" and col !="customerID"):
        categoric_variables.append(col)

# lets use pd.getdummies for categoric_variables 

df = pd.get_dummies(df,prefix = categoric_variables,columns = categoric_variables)


ax = sns.heatmap(df.corr(),annot=True,cmap="YlGnBu",xticklabels=1,yticklabels=1,fmt=".2f")
plt.title("Correlations Between Variables")


# train test split
# before train_test_split, we will use tenure_interval for classification, lets delete tenure feature

df.drop("tenure",axis=1,inplace=True)

# customer_id ->index
df.set_index("customerID",inplace=True)

from sklearn.model_selection import train_test_split
# create dependent and independet variable

X = df.drop(["Churn"],axis=1)
y = df[["Churn"]]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=42,stratify=y)

# Normalize MontlyCharges and TotalCharges values based on train set mean and std to prevent information linkage

MonthlyCharge_train_mean = np.mean(X_train["MonthlyCharges"])
MonthlyCharge_train_std = np.std(X_train["MonthlyCharges"])

TotalCharge_train_mean = np.mean(X_train["TotalCharges"])
TotalCharge_train_std = np.std(X_train["TotalCharges"])

# Min-max normalization: Guarantees all features will have the exact same scale but does not handle outliers well.
# Z-score normalization: Handles outliers, but does not produce normalized data with the exact same scale.
# lets try z-score

# Z-score  normalization
X_train["MonthlyCharges"] = X_train["MonthlyCharges"].apply(lambda x: (x-MonthlyCharge_train_mean)/MonthlyCharge_train_std)

X_test["MonthlyCharges"] = X_test["MonthlyCharges"].apply(lambda x: (x-MonthlyCharge_train_mean)/MonthlyCharge_train_std)


X_train["TotalCharges"] = X_train["TotalCharges"].apply(lambda x: (x-TotalCharge_train_mean)/TotalCharge_train_std)

X_test["TotalCharges"] = X_test["TotalCharges"].apply(lambda x: (x-TotalCharge_train_mean)/TotalCharge_train_std)




    
# Classification
# import some libraries for classification
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from yellowbrick.classifier import DiscriminationThreshold
from yellowbrick.classifier import ROCAUC


def evaluate_model(model,x_test,y_test,coef_):
    
    prediction = model.predict(x_test)
    acc_test = accuracy_score(y_test,prediction)
    acc_train = accuracy_score(y_train,model.predict(X_train))
    print("\n")
    print("Accuracy Score(Test): " + str(acc_test))
    print("Accuracy Score(Train): " + str(acc_train))
    print("Difference between train and test accuracy = {0}".format(abs(acc_test-acc_train)))
    print("Roc Auc Score: "+ str(roc_auc_score(y_test,prediction)))
    print("\n")
    print("Classification Report:")
    print(classification_report(y_test,prediction))
    # confusion matrix
    plt.figure()
    cm = confusion_matrix(y_test,prediction)
    sns.heatmap(cm,annot=True,cmap="YlGnBu",fmt="d")
    plt.title("Confusion Matrix(1:Churned, 0:Not Churned)")
    plt.show()
    
    # roc-curve
    plt.figure()
    visualizer = ROCAUC(model, classes=["Not Churn", "Churn"])
    
    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()                       # Finalize and show the figure    
    plt.show()
    
    
    if coef_:
        feature_imp ={}
        
        for idx,col_name in enumerate(X_train.columns):
            feature_imp[col_name] = model.coef_[0][idx]
            
        
        feature_imp = pd.DataFrame(feature_imp.items(),columns = ["Feature","Feature Importance"])
        feature_imp.set_index("Feature",inplace=True)
        
        
        ax = feature_imp.plot(kind="bar",fontsize=10,color="red")
        
        ax.set_title("Future Importance",fontdict={"fontsize":12,"fontweight":"bold"})
        ax.set_ylabel("Coef_")
        
        plt.show()
        

def evaluate_ANN(prediction,y_test,pred_train,y_train):
    
    acc_test = accuracy_score(y_test,prediction)
    acc_train = accuracy_score(y_train,pred_train)
    print("\n")
    print("Accuracy Score(Test): " + str(acc_test))
    print("Accuracy Score(Train): " + str(acc_train))
    print("Difference between train and test accuracy = {0}".format(abs(acc_test-acc_train)))
    print("Roc Auc Score: "+ str(roc_auc_score(y_test,prediction)))
    print("\n")
    print("Classification Report:")
    print(classification_report(y_test,prediction))
    # confusion matrix
    plt.figure()
    cm = confusion_matrix(y_test,prediction)
    sns.heatmap(cm,annot=True,cmap="YlGnBu",fmt="d")
    plt.title("Confusion Matrix(1:Churned, 0:Not Churned)")
    plt.show()
    
    
    

# LOGISTIC REGRESSION
    
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(X_train,y_train)


evaluate_model(LR,X_test,y_test,True) # evaluate model

# let's look discimination 

pred = LR.predict(X_test)
proba = LR.predict_proba(X_test) # discrimination threshold is 0.5, let's find best disc. threshhold.
proba = pd.DataFrame(proba,columns=["0","1"])
proba["Selected Class"] = pred

# try to best threshold to maximize f1 score
vis = DiscriminationThreshold(LR)
vis.fit(X_train,y_train)
vis.poof()  # algorithm trys to maximize f1 score  
# threshold = 0.29

# KNN
from sklearn.neighbors import KNeighborsClassifier


k_scores = {}
for k in range(1,30,2):
    
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(X_train,y_train)

    k_scores[k] = [KNN.score(X_test,y_test),roc_auc_score(y_test,KNN.predict(X_test))]
    
    
k_scores = pd.DataFrame(k_scores.items(),columns=["k","Accuracy"])

k_scores["Roc Score"] = k_scores["Accuracy"].apply(lambda x: x[1])

k_scores["Accuracy"] = k_scores["Accuracy"].apply(lambda x: x[0])

k_scores.set_index("k",inplace=True)

# plot graph
ax = k_scores.plot()
ax.set_title("ROC/ACC scores for different k paramaters",fontdict={"fontsize":12,"fontweight":"bold"})


# k = 9 looks good

# use minkowski and p = 2  (euclidian distance) as ditance metric

KNN = KNeighborsClassifier(n_neighbors=9)
KNN.fit(X_train,y_train)
evaluate_model(KNN,X_test,y_test,False) # evaluate model


pred = KNN.predict(X_test)
proba = KNN.predict_proba(X_test)
proba = pd.DataFrame(proba,columns=["0","1"])
proba["Selected Class"] = pred

# try to best threshold to maximize f1 score
vis = DiscriminationThreshold(KNN)
vis.fit(X_train,y_train)
vis.poof()  # algorithm trys to maximize f1 score  
# threshold = 0.30

# NAIVE BAYES

# GAUSTTIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()
GNB.fit(X_train,y_train)

evaluate_model(GNB,X_test,y_test,False)



pred = GNB.predict(X_test)
proba = GNB.predict_proba(X_test)
proba = pd.DataFrame(proba,columns=["0","1"])
proba["Selected Class"] = pred

vis = DiscriminationThreshold(GNB)
vis.fit(X_train,y_train)
vis.poof()  # algorithm trys to maximize f1 score  
# threshold = 0.89

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

n_estimators = {}

for n in range(10,160,10):
    
    RF = RandomForestClassifier(n_estimators=n)
    RF.fit(X_train,y_train)

    n_estimators[n] = [RF.score(X_test,y_test),roc_auc_score(y_test,RF.predict(X_test))]


n_estimators = pd.DataFrame(n_estimators.items(),columns=["n","Accuracy"])

n_estimators["Roc Score"] = n_estimators["Accuracy"].apply(lambda x: x[1])

n_estimators["Accuracy"] = n_estimators["Accuracy"].apply(lambda x: x[0])

n_estimators.set_index("n",inplace=True)

ax = n_estimators.plot()
ax.set_title("ROC/ACC scores for different n_estimators paramaters",fontdict={"fontsize":12,"fontweight":"bold"})

# n_estimators = 60 looks good 

# let's try gini and entropy

RF = RandomForestClassifier(n_estimators = 60,criterion="gini")

RF.fit(X_train,y_train)

evaluate_model(RF,X_test,y_test,False) # overfit, let's limit max depth


RF = RandomForestClassifier(n_estimators = 60,criterion="gini",max_depth=6)

RF.fit(X_train,y_train)

evaluate_model(RF,X_test,y_test,False) 

# Random Forest Feature Importance
feature_imp ={}

for idx,col_name in enumerate(X_train.columns):
    feature_imp[col_name] = RF.feature_importances_[idx]
    

feature_imp = pd.DataFrame(feature_imp.items(),columns = ["Feature","Feature Importance"])
feature_imp.set_index("Feature",inplace=True)


ax = feature_imp.plot(kind="barh",fontsize=10,color="red")

ax.set_title("Future Importance",fontdict={"fontsize":12,"fontweight":"bold"})

plt.show()


# XGBOOST

from xgboost import XGBClassifier

# let's find best n_est. paramater
n_estimators = {}

for n in range(10,160,10):
    
    XGB = XGBClassifier(n_estimators=n)
    XGB.fit(X_train,y_train)

    n_estimators[n] = [XGB.score(X_test,y_test),roc_auc_score(y_test,XGB.predict(X_test))]


n_estimators = pd.DataFrame(n_estimators.items(),columns=["n","Accuracy"])

n_estimators["Roc Score"] = n_estimators["Accuracy"].apply(lambda x: x[1])

n_estimators["Accuracy"] = n_estimators["Accuracy"].apply(lambda x: x[0])

n_estimators.set_index("n",inplace=True)

ax = n_estimators.plot()
ax.set_title("ROC/ACC scores for different n_estimators paramaters",fontdict={"fontsize":12,"fontweight":"bold"})

# n_estimators = 30 looks good

XGB = XGBClassifier(n_estimators=30)

XGB.fit(X_train,y_train)

evaluate_model(XGB,X_test,y_test,False) # overfit, let's reduce leraning rate

XGB = XGBClassifier(n_estimators=30,learning_rate = 0.1,random_state=42,max_depth=5)

XGB.fit(X_train,y_train)

evaluate_model(XGB,X_test,y_test,False)

# Feature Importance for XGBoost

feature_imp ={}

for idx,col_name in enumerate(X_train.columns):
    feature_imp[col_name] = XGB.feature_importances_[idx]
    

feature_imp = pd.DataFrame(feature_imp.items(),columns = ["Feature","Feature Importance"])
feature_imp.set_index("Feature",inplace=True)


ax = feature_imp.plot(kind="barh",fontsize=10,color="red")

ax.set_title("Future Importance",fontdict={"fontsize":12,"fontweight":"bold"})

plt.show()

# NEURAL NETWORK

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU



model = Sequential()

model.add(Dense(16,kernel_initializer = "uniform",activation = LeakyReLU(),input_dim = 32))


model.add(Dense(16,kernel_initializer = "uniform",activation = LeakyReLU()))

model.add(Dense(1,kernel_initializer = "uniform",activation = "sigmoid"))

model.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])


model.summary()


history = model.fit(X_train,y_train,batch_size = 16,epochs=10)


plt.figure()
plt.plot(history.history["accuracy"],c="red")
plt.title("Accuracy-Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.figure()
plt.plot(history.history["loss"],c="green")
plt.title("Loss-Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 8 looks good, let's train ANN with 8 epoch

model = Sequential()

model.add(Dense(16,kernel_initializer = "uniform",activation = LeakyReLU(),input_dim = 32))


model.add(Dense(16,kernel_initializer = "uniform",activation = LeakyReLU()))

model.add(Dense(1,kernel_initializer = "uniform",activation = "sigmoid"))

model.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

history = model.fit(X_train,y_train,batch_size = 16,epochs=8)


pred = pd.DataFrame(model.predict(X_test),index = X_test.index,columns=["Prediction"])

# let's find best threshold to maximize f1 score

def find_ANN_f1_score(prediction,real,th):
    
    preds = []
    for i in prediction:
        if i>=th:
            preds.append(1)
        else:
            preds.append(0)
    
    return f1_score(real,preds),accuracy_score(real,preds)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

th_f1 = {}

for t in thresholds:
    th_f1[t] = find_ANN_f1_score(model.predict(X_test),y_test,t)


# let's plot result's
    
th_f1 = pd.DataFrame(th_f1.items(),columns=["threshold","f1-score"])

th_f1.set_index("threshold",inplace=True)

th_f1["accuracy"] = th_f1["f1-score"].apply(lambda x: x[1])

th_f1["f1-score"] = th_f1["f1-score"].apply(lambda x: x[0])

th_f1.plot()
plt.title("Accuracy and f1 score by different threshold values")
plt.ylabel("Acc and f1-score")

# 0.4 looks good for threshold

# test predictions
proba_test = model.predict(X_test)
predictions_test = []

for i in proba_test:
    if i >=0.4:
        predictions_test.append(1)
    else:
        predictions_test.append(0)
        
# train predictions
proba_train = model.predict(X_train)
predictions_train = []

for i in proba_train:
    if i >=0.4:
        predictions_train.append(1)
    else:
        predictions_train.append(0)
        
evaluate_ANN(predictions_test,y_test,predictions_train,y_train)


# COMPARİNG CLASSIFICATION MODELS 
# dataset is inbalanced, so used f1 score rather than precision and recall. it will give better information

results = pd.DataFrame(columns=["Model","Train Acc","Test Acc","f1-score","ROC-AUC score"])
    
results = results.append({"Model":"Logistic Regression","Train Acc":0.8060,"Test Acc":0.8021,"f1-score":0.5721,"ROC-AUC score":0.7050},ignore_index=True)
 
results = results.append({"Model":"KNN","Train Acc":0.8187,"Test Acc":0.7862,"f1-score":0.5664,"ROC-AUC score":0.7030},ignore_index=True)
  
results = results.append({"Model":"Gaussian NB","Train Acc":0.7467,"Test Acc":0.7462,"f1-score": 0.6193,"ROC-AUC score":0.7560},ignore_index=True)
 

results = results.append({"Model":"Random Forest","Train Acc":0.8083,"Test Acc":0.7913,"f1-score": 0.5106,"ROC-AUC score":0.6696},ignore_index=True)

results = results.append({"Model":"XGBoost","Train Acc":0.8229,"Test Acc":0.8038,"f1-score":  0.5835,"ROC-AUC score": 0.7123},ignore_index=True)


results = results.append({"Model":"ANN","Train Acc":0.7896,"Test Acc":0.7862,"f1-score":  0.6231,"ROC-AUC score": 0.7476},ignore_index=True) 






   

   




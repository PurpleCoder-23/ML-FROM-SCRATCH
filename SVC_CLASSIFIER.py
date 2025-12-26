# IMPLEMENTING SUPPORT VECTOR MACHINE CLASSIFIER FROM SCRATCH USING PYTHON
# STEP1:LOAD THE DATASET
# STEP2:PERFORM BASIC EXPLORATORY DATA ANALYSIS
# STEP3:PREPARE DATA
# STEP4:DEFINE HINGE LOSS
# STEP5:DEFINE GRADIENT DESCENT
# STEP6:TRAIN SVC MODEL
# STEP7:DEFINE PREDICTION FUNCTION
# STEP8:IMPLEMENT CONFUSION MATRIX
# STEP9:IMPLEMENT ACCURACY, PRECISION , RECALL , F1_SCORE
# STEP10:EVALUATE SCRATCH SVC
# STEP11:TRAIN SKLEARN SVC
# STEP12:COMPARE RESULTS(SKLEARN VS SCRATCH SVC)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#STEP 1: LOAD THE DATASET
df=pd.read_csv(r"C:\Users\HOME\Desktop\ML_ALGO\diabetes.csv")
print(df.head(5))
#STEP2:PERFORM BASIC EDA

#get data inforamtion
print(df.info())
#get statistical summary
print(df.describe())
#checking missing or zero values
zero_cols=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[zero_cols]=df[zero_cols].replace(0,np.nan)
print(df.isnull().sum())
#Handle Missing Values(Median Imputation)
df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())
#target variable Distribution
sns.countplot(x="Outcome",data=df)
plt.title("Diabetes Outcome Distribution")
plt.show()
#Feature Distributions
df.hist(figsize=(15,10))
plt.suptitle("Feature Distribution")
plt.show()
#correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


#STEP 3:PREPARE DATA

#separate Features and Target
X=df.drop("Outcome",axis=1).values
Y=df["Outcome"].values

#convert labels for svm(0to-1,1to+1)
Y=np.where(Y==0,-1,1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)


#STEP4:DEFINE THE HINGE LOSS FUNCTION
#lamda_parameter is regularization parameter
#loss=max(0,1-y(w.x+b))
#hinge loss (maximize margin or incur penalty)

def hinge_loss(X,Y,w,b,lambda_parameter):
    distances=1-Y*(np.dot(X,w)+b)
    distances=np.maximum(0,distances)
    # Standard Soft-Margin SVM formula: C * sum(violations) + 1/2*||w||^2
    return np.mean(distances) + lambda_parameter* np.dot(w, w)


#STEP 5:DEFINE GRADIENT DESCENT FUNCTION
#m rows no. of observations
#n cols no. of independent features
def gradient_descent_step(X,Y,w,b,learning_rate,lambda_parameter):
    m,n=X.shape
    for i in range(m):
        condition=Y[i]*(np.dot(X[i],w) + b)>=1
        if condition:
            # Only the regularization part (2 * 0.5 * w = w)
            # We want to keep weights small to maximize margin
            dw = 2 * lambda_parameter* w 
            db = 0
        else:
            # Regularization part + Misclassification part
            # Gradient of [1 - y(wx+b)] with respect to w is (-y * x)
            dw = 2 * lambda_parameter * w - np.dot(X[i], Y[i])
            db = -Y[i]
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate* db
        
    return w, b

# STEP6: TRAIN SVC SCRATCH MODEL
w=np.zeros(X.shape[1])
b=0.0

learning_rate=0.001
lambda_parameter=0.01
epochs=700

loss_history=[]

for epoch in range(epochs):
    loss=hinge_loss(X,Y,w,b,lambda_parameter)
    loss_history.append(loss)
    w,b=gradient_descent_step(X,Y,w,b,learning_rate,lambda_parameter)

print("Final Weights:",w)
print("Final Bias:",b)


#STEP 7:DEFINE PREDICTION FUNCTION
#outcome could be -1 or 1
def predict(X,w,b):
    return np.sign(np.dot(X,w)+b)


#STEP 8:IMPLEMENT CONFUSION MATRIX
def confusion_matrix(Y,y_pred):
    TP=np.sum((Y==1) & (y_pred==1))
    TN=np.sum((Y==-1) & (y_pred==-1))
    FP=np.sum((Y==-1) & (y_pred==1))
    FN=np.sum((Y==1) & (y_pred==-1))
    return TP,TN,FP,FN

#STEP 9:IMPLEMENT EVALUATION METRICS
def accuracy(TP,TN,FP,FN):
    return (TP+TN)/(TP+TN+FP+FN)

def precision(TP,FP):
    return TP/(TP+FP) if TP+FP!=0 else 0

def recall(TP,FN):
    return TP/(TP+FN) if TP+FN!=0 else 0

def f1_score(p,r):
    return 2*p*r/(p+r) if p+r!=0 else 0


#STEP 10: EVALUATE SCRATCH SVC
y_pred=predict(X,w,b)

TP,TN,FP,FN=confusion_matrix(Y,y_pred)

acc_scratch=accuracy(TP,TN,FP,FN)
prec_scratch=precision(TP,FP)
rec_scratch=recall(TP,FN)
f1_scratch=f1_score(prec_scratch,rec_scratch)

print("\nSUPPOR VECTOR CLASSIFIER SCRATCH MODEL PERFORMANCE")
print(f"Accuracy : {acc_scratch:.2f}")
print(f"Precision : {prec_scratch:.2f}")
print(f"Recall : {rec_scratch:.2f}")
print(f"F1 Score : {f1_scratch:.2f}")


#STEP 11: TRAIN SKLEARN SVC

from sklearn.svm import SVC
sk_model=SVC(kernel="linear")
sk_model.fit(X,Y)

y_pred_sk=sk_model.predict(X)

#STEP12: COMPARE RESULTS (SKLEARN VS SCRATCH)
TP_s, TN_s, FP_s, FN_s = confusion_matrix(Y, y_pred_sk)

acc_sk = accuracy(TP_s, TN_s, FP_s, FN_s)
prec_sk = precision(TP_s, FP_s)
rec_sk = recall(TP_s, FN_s)
f1_sk = f1_score(prec_sk, rec_sk)

print("\nMODEL COMPARISON")
print("-" * 45)
print("Metric        Scratch SVC    Sklearn SVC")
print(f"Accuracy      {acc_scratch:.2f}          {acc_sk:.2f}")
print(f"Precision     {prec_scratch:.2f}          {prec_sk:.2f}")
print(f"Recall        {rec_scratch:.2f}          {rec_sk:.2f}")
print(f"F1 Score      {f1_scratch:.2f}          {f1_sk:.2f}")

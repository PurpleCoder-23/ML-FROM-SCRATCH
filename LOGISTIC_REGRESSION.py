#IMPLEMENTING LOGISTIC REGRESSION FROM SCRATCH USING PYTHON
#STEP1:CREATE A BINARY CLASSIFICATION DATASET
#STEP2:VISUALIZE THE DATA
#STEP3:DEFINE SIGMOID FUNCTION
#STEP4:DEFINE LOG LOSS (BINARY CROSS ENTROPY LOSS FUNCTION)
#STEP5:DEFINE GRADIENT DESCENT
#STEP6:TRAIN THE MODEL
#STEP7:MAKING PREDICTIONS
#STEP8:IMPLEMENT CONFUSION MATRIX
#STEP9:IMPLEMENT ACCURACY,PRECISION,RECALL,F1-SCORE
#STEP10:COMPARE WITH SKLEARN PREBUILT LOGISTIC REGRESSION
#STEP11:TRAIN SKLEARN LOGISTIC REGRESSION
#STEP12:COMPARE RESULTS(SKLEARN VS SCRATCH MODEL)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#STEP1:CREATE DATASET
df = pd.read_excel("dog_cat_features_dataset.xlsx")

# Encode target
df["Target"] = df["Target"].map({"Cat": 0, "Dog": 1})

X = df.drop("Target", axis=1).values
Y = df["Target"].values

#STEP2:Visualize dataset

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="bwr")
plt.xlabel("Weight (kg)")
plt.ylabel("Height (cm)")
plt.title("Dog vs Cat")
plt.show()


#STEP3:SIGMOID FUNCTION
#z=np.dot(X,w)+b
def sigmoid(z):
    return 1/(1+np.exp(-z))



#STEP4:LOGLOSS FUNCTION
#epsilon is small value as log(0) is undefined and log(1) is 0 ,here np.clip dfines the range
#y_pred < epsilon → use epsilon and y_pred > 1 - epsilon → use 1 - epsilon

def log_loss(Y,y_pred):
    epsilon=1e-9
    y_pred=np.clip(y_pred,epsilon,1-epsilon)
    return -np.mean(Y*np.log(y_pred)+(1-Y)*np.log(1-y_pred))


#STEP5:GRADIENT DESCENT STEP
def gradient_descent_step(X,Y,w,b,learning_rate):
    """
    Docstring for gradient_descent_step
    
    param X: (m,n) feature matrix
    param Y: (m,) target vector
    param w: (n,) weight vector
    param b: scalar bias
    param learning_rate: stepsize at each iteration
    """
    #number of data points(rows) and features(columns)
    m,n=X.shape

    z=np.dot(X,w)+b

    y_pred=sigmoid(z)

    error=y_pred-Y

    #calculating gradients
    dw=(1/m)*np.dot(X.T,error)
    db=(1/m)*np.sum(error)
    
    #Updating weights and bias
    w-=learning_rate*dw
    b-=learning_rate*db

    return w,b


 #STEP6:TRAINING THE LOOP
w=np.zeros(X.shape[1])
b=0.0
learning_rate=0.1
epochs=500

loss_history=[]

for epoch in range(epochs):
    z=np.dot(X,w)+b
    y_pred=sigmoid(z)

    loss=log_loss(Y,y_pred)
    loss_history.append(loss)

    w,b=gradient_descent_step(X,Y,w,b,learning_rate)

print(f"Final weights:{w}")

print(f"Final bias:{b:.4f}")


#STEP 7:PREDICTION FUNCTION
def predict(X,w,b,threshold=0.5):
    z=np.dot(X,w)+b
    y_pred=sigmoid(z)
    return (y_pred>=threshold).astype(int).flatten()

#STEP8:CONFUSION MATRIX

def confusion_matrix(Y,y_pred):
    TP=np.sum((Y==1)&(y_pred==1))
    TN=np.sum((Y==0)&(y_pred==0))
    FP=np.sum((Y==0)&(y_pred==1))
    FN=np.sum((Y==1)&(y_pred==0))
    return TP,TN,FP,FN


#STEP 9:COMPUTING METRICS
def accuracy(TP,TN,FP,FN):
    return (TP+TN)/(TP+TN+FP+FN)

def precision(TP,FP):
    return TP/(TP+FP) if TP+FP!=0 else 0

def recall(TP,FN):
    return TP/(TP+FN) if TP+FN!=0 else 0

def f1_score(p,r):
    return 2*p*r/(p+r) if p+r!=0 else 0


#STEP 10:EVALUATION

y_pred=predict(X,w,b)
TP,TN,FP,FN=confusion_matrix(Y,y_pred)

acc_scratch=accuracy(TP,TN,FP,FN)
prec_scratch=precision(TP,FP)
rec_scratch=recall(TP,FN)
f1_scratch=f1_score(prec_scratch,rec_scratch)


print("\nEvaluation Metrics:")
print(f"Accuracy : {acc_scratch:.2f}")
print(f"Precision : {prec_scratch:.2f}")
print(f"Recall : {rec_scratch:.2f}")
print(f"F1 Score : {f1_scratch:.2f}")




#STEP11:SKLEARN LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
sk_model=LogisticRegression()
sk_model.fit(X,Y)
y_pred_sklearn=sk_model.predict(X)
TP_s,TN_s,FP_s,FN_s=confusion_matrix(Y,y_pred_sklearn)

acc_sk=accuracy(TP_s,TN_s,FP_s,FN_s)
prec_sk=precision(TP_s,FP_s)
rec_sk=recall(TP_s,FN_s)
f1_sk=f1_score(prec_sk,rec_sk)


#STEP12:RESULT COMPARISON
print("\nMODEL COMPARISON")
print("-" * 40)
print("Metric        Scratch     sklearn")
print(f"Accuracy      {acc_scratch:.2f}        {acc_sk:.2f}")
print(f"Precision     {prec_scratch:.2f}        {prec_sk:.2f}")
print(f"Recall        {rec_scratch:.2f}        {rec_sk:.2f}")
print(f"F1 Score      {f1_scratch:.2f}        {f1_sk:.2f}")

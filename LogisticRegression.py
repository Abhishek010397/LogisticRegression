import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

data_df= pd.read_csv('Questionnaire-response.csv')
data_df = data_df.iloc[1: , :]

X= data_df[['Q2','Q4','Q5']].values
Y = (data_df['Q8'] == 'YES').astype(np.int32).values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3,random_state=0)


### Train Logistic Regression
model = LogisticRegression()
model.fit(X_train,Y_train)
prediction_Y = model.predict(X_test)

## MODEL EVALUATION
print('Accuracy',metrics.accuracy_score(Y_test,prediction_Y))
Y_Pred_Proba = model.predict_proba(X_test)[:,1]
auc = metrics.roc_auc_score(Y_test,Y_Pred_Proba)
print('AUC:', round(auc,2))


##PREDICTION
prediction = model.predict(([[3,4,4]]))
if prediction == 0:
    print('NO')
if prediction ==1:
    print('YES')


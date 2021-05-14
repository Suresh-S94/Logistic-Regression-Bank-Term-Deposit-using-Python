import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("E:\\Data Science_Excelr\\Logistic Regression\\bank-full.csv")
print(df.head())

df1 =  pd.read_csv('E:\\Data Science_Excelr\\Logistic Regression\\bank-full.csv', sep=';')
df1.head()

df2=pd.DataFrame(df1)
df2.head()

df2.drop(["age","job","marital","education","contact","day","month"], axis=1, inplace=True)
df2.head(3)

df2 = pd.get_dummies(data=df2, columns=["default","housing","loan","poutcome"])
df2['Y'] = df2.y.map({'no':0,'yes':1})
df2.drop(["y"],axis=1, inplace=True)
df2.head()

df2.iloc[:,:].isnull().values.any()

df2 = df2[['Y','balance','duration','campaign','pdays','previous','default_no','default_yes','housing_no','housing_yes','loan_no','loan_yes','poutcome_failure','poutcome_other','poutcome_success','poutcome_unknown']]
df2.head(2)

from sklearn.linear_model import LogisticRegression

x=df2.iloc[:,1:]
y=df2.iloc[:,0]

classifier = LogisticRegression()
classifier.fit(x,y)

classifier.score(x,y)
print("Model score : ",classifier.score(x,y))
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
confusion_matrix(classifier.predict(x),y)
print("confusion matrix :  ""\n",confusion_matrix(classifier.predict(x),y))

from sklearn.metrics import classification_report
print(classification_report(y,classifier.predict(x)))

FalsePositive, TruePositive,_=roc_curve(classifier.predict(x),y,drop_intermediate=False)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(FalsePositive, TruePositive, color='red',label='ROC curve')
plt.plot([0, 1], [0, 1], color='blue',linestyle='--')
plt.xlabel('FalsePositive')
plt.ylabel('TruePositive')
plt.title('ROC curve')
plt.show()

y_pred = classifier.predict(x)
df1["yy"] = y_pred
df1['yy'] = df1.yy.map({0:'no',1:'yes'})
df1.head()

y_prob = pd.DataFrame(classifier.predict_proba(x.iloc[:,:]))  
print(y_prob)

new_df = pd.concat([df1,y_prob], axis=1)
new_df.head()

new_df.to_csv("Predcited Term deposit.csv")

pred=pd.read_csv('Predcited Term deposit.csv')

pred.head()


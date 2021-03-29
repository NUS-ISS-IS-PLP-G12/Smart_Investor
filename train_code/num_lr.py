import pandas as pd
from sklearn.preprocessing import StandardScaler
num=pd.read_table("num2.csv",names = ["open", "high","low","volume","label"])
num
X = num.iloc[:,0:4]

X_train = num.iloc[0:1391,0:4]
X_test  = num.iloc[1391:1988,0:4]
y = num.loc[:,'label']

y_train= num.iloc[0:1391,4]
y_test= num.iloc[1391:1988,4]
print(y_test)
from sklearn.linear_model import LogisticRegression

# fit final model
model = LogisticRegression()
model.fit(X_train, y_train)
# new instances where we do not know the answer
# make a prediction
ynew = model.predict_proba(X_test)
print(ynew)
# import pickle 
import joblib
from sklearn.svm import SVC
from sklearn import datasets
import pickle as p
with open('num_lr.pickle','wb') as fw:
    p.dump(model,fw)
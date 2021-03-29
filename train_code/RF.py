

from sklearn.ensemble import RandomForestClassifier
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
clf = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=0)
classifier = clf.fit(X_train,y_train)
# import pickle 
import joblib
from sklearn.svm import SVC
from sklearn import datasets
import pickle as p
with open('num_RF.pickle','wb') as fw:
    p.dump(classifier,fw)
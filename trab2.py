import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from matplotlib import pyplot as plt

df = pd.read_csv('leaf.csv')

x = df.iloc[:,2:16]
y = df.iloc[:,0]

sc = StandardScaler()  
x = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

clf=RandomForestClassifier(n_estimators=100)
# print(cross_val_predict(clf, x, y, cv=10))

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
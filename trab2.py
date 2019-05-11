import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import matplotlib.pyplot as plt

df = pd.read_csv('leaf.csv')

x = df.iloc[:,2:16]
y = df.iloc[:,0]

sc = StandardScaler()  
x = sc.fit_transform(x)
clf=RandomForestClassifier(n_estimators=100)

# treina o modelo com 80% dos dados, prediz o de 20% (dados teste) e cacula acurácia 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# faz o kfold, treina e prediz para cada fold, calcula o score e retorna um array de todos os scores.
kf = StratifiedKFold(n_splits=10)
scores = cross_val_score(clf, x, y, cv=kf)
print("Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

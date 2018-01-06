import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
X=titanic[[
'pclass','age','sex'
]]
y=titanic['survived']

X['age'].fillna(X['age'].mean(),inplace=True)

from sklearn.cross_validation import  train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

from sklearn.feature_extraction import  DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_predict = dtc.predict(X_test)


from sklearn.ensemble import  RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred = rfc.predict(X_test)


from sklearn.ensemble import  GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred = gbc.predict(X_test)


from sklearn.metrics import  classification_report
print 'The accuracy of decision tree is :',dtc.score(X_test,y_test)
print classification_report(dtc_y_predict,y_test)

print 'The accuracy of random forset classifier is ',rfc.score(X_test,y_test)
print classification_report(rfc_y_pred,y_test)

print 'The accuracy of gradiet tree boosting is',gbc.score(X_test,y_test)
print classification_report(gbc_y_pred,y_test)
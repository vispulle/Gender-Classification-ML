from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
clf1 = tree.DecisionTreeClassifier()
clf2 = svm.SVC()
clf3 = KNeighborsClassifier()

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


clf1 = clf1.fit(X,Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)
prediction = clf1.predict([[190,70,43]])
prediction2 = clf2.predict([[190,70,43]])
prediction3 = clf3.predict([[190,70,43]])

print (prediction3,prediction2,prediction)

predict_dt = clf1.predict(X)
predict_svm = clf2.predict(X)
predict_knn = clf3.predict(X)

accuracy_dt = accuracy_score(Y,predict_dt)
accuracy_svm = accuracy_score(Y,predict_svm)
accuracy_knn = accuracy_score(Y,predict_knn)


print(np.max([accuracy_dt,accuracy_knn,accuracy_svm]))

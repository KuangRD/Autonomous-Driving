## SVM

```sh
from sklearn.svm import SVC
clf = SVC(kernel="linear")
# SVC == Support Vector Classifier
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

accuracy = clf.score(features_test,labels_test)
```
常用核函数`rbf`, `sigmoid`, `poly` 和`linear`

**Tips:** 线性核函数只能调参数`C`,非线性核函数可以调`C`和`gamma`

### Parameter Tuning 调参
[GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) 用3折法遍历验证参数空间的每种可能

[RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV) 随机验证参数空间的若干种可能


```sh
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
```

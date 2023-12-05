from typing import List

import sklearn as skl
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y, test_size=0.1)

# print(x_train)
# print(y_train)

classes: list[str] = ['Malignant', 'Benign']

# instantiate Support Vector Classifier
clf = svm.SVC(kernel='linear', C=2)
# assign data to classifier
clf.fit(x_train, y_train)

prediction = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, prediction)

# labels for binary output
definition = ['Malignant', 'Benign']

# print(accuracy)

# show iterated output
for i in range(len(prediction)):
    print(definition[prediction[i]], definition[y_test[i]])

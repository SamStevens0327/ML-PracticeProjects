import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np

data = pd.read_csv('car.data')

# labels need to assigned to integer values
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maintenance = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
clss = le.fit_transform(list(data['class']))

predict = 'class'

# list of tuples containing one of each data variable with corresponding indexes
x = list(zip(buying, maintenance, door, persons, lug_boot, safety))
# variable to be predicted in list form
y = list(clss)

# split data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# troubleshooting for optimal accuracy
best = 0
k = list(range(1, 21, 2))
for i in k:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    if accuracy > best:
        best = accuracy
accuracy = best

predicted = model.predict(x_test)

names = ['unacc', 'good', 'acc', 'vgood']

for i in range(len(predicted)):
    print('Predicted: ', names[predicted[i]], 'Actual:', names[y_test[i]])
    n = model.kneighbors(x_test, 9, True)
    assert isinstance(n, object)
    print(n)

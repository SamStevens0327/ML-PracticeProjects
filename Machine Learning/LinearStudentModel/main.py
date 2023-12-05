import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('student-mat.csv', sep=';')
data = data[
    ['G1', 'G2', 'G3', 'studytime', 'failures', 'absences', 'higher', 'famsup']
    [:]
]
# [columns][rows]

data['higher'] = data['higher'].map({'yes': 1, 'no': 0})
data['famsup'] = data['famsup'].map({'yes': 1, 'no': 0})
# assigns binary value to yes/no

x = np.array(data.drop(['G3'], axis=1))  # 1 = drop said columns, 0 = drop said row (labels)
y = np.array(data['G3'])  # exclusively 'G3', therefore y = inverse of x

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# loop to find and save highest accuracy model, then deactivate loop to work with saved model
# best = 0
# for _ in range(99):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#     # split into train/test data
#
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)  # scoring the model to instantiate the accuracy
#
#     if accuracy > best:
#         best = accuracy
#         print(accuracy)
#         with open('student_model.pickle','wb') as file:  # writes/save in binary form 'pickle file' in directory
#             pickle.dump(linear, file)

# after loop is deactivated, load the saved high accuracy model
pickle_in = open('student_model.pickle', 'rb')  # reads in binary form the previously saved file

loaded_model = pickle.load(pickle_in)  # Load the model from the pickle file

# printprint('coefficient: \n', linear.coef_)
# print('intercept: \n', linear.intercept_)
# view model parameters

predictions = loaded_model.predict(x_test)  # using data that wasn't trained on to make predictions

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])  # prints the; predicted value, input data array, actual answer

# plotting data
x_variable = 'G1'
plt.style.use('ggplot') # so it looks pretty
plt.scatter(data[x_variable], data['G3'])
plt.xlabel(x_variable)
plt.ylabel('Final Grade')
plt.show()

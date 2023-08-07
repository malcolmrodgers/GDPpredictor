import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn import linear_model
import pickle
import matplotlib.pyplot as pyplot


data = pd.read_csv('/Users/malcolmrodgers/Documents/Coding/Python/GDPpredictor/worlddata2023clean.csv', sep=',')
print(data.head())

#value we want to predict
predict = 'GDP'

#separate predict column and store both in numpy arrays
x = np.array(data.loc[:, 'Armed Forces size':'Latitude'])
y = np.array(data[predict])

highscore = 0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''
for k in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)

    if acc > highscore:
        highscore = acc
        with open('GDPPredictor.pickle', 'wb') as f:
            pickle.dump(model, f)
'''

pickle_in = open("/Users/malcolmrodgers/Documents/Coding/Python/GDPpredictor/GDPPredictor.pickle", "rb")
model = pickle.load(pickle_in)



print(highscore)





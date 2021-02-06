import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine


data = pd.read_csv('/Users/alliewu/Desktop/Chevron/filesForStartOfDatathon/training.csv')

# store1
store1 = data[data['StoreNumber']==1000]

le = preprocessing.LabelEncoder()
# storenum = data['StoreNumber']
storenum = le.fit_transform(list(data['StoreNumber']))
dayOfTheYear = data['dayOfTheYear']
hrbucket = data['3HourBucket']
# hrbucket = le.fit_transform(list(data['3HourBucket']))
ebt = le.fit_transform(list(data['EBT Site']))
loyalty = le.fit_transform(list(data['Loyalty Site']))
extra = le.fit_transform(list(data['ExtraMile Site']))
cobrand = le.fit_transform(list(data['CoBrand']))
alc = le.fit_transform(list(data['Alcohol']))
cwash = le.fit_transform(list(data['Carwash']))
food = le.fit_transform(list(data['Food Service']))
city = le.fit_transform(list(data['City']))
state = le.fit_transform(list(data['State']))

GrossSoldQuantity = le.fit_transform(data['GrossSoldQuantity'])


x = list(zip(storenum, dayOfTheYear, hrbucket, ebt, loyalty, extra, cobrand, alc, cwash, food, city)) #features
y = list(GrossSoldQuantity) #labels

# accuracy = 0
# index = 0
# for i in range(100):
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=88)

pipe = make_pipeline(StandardScaler(), linear_model.LinearRegression())
pipe.fit(x_train, y_train)
acc = pipe.score(x_test, y_test)

    # if accuracy < acc:
    #     accuracy = acc
    #     index = i
# print(accuracy, index)

predicted = pipe.predict(x_test)
for x in range(len(predicted)):
    print("Data:", x_test[x], "Predicted:", int(predicted[x]), "Actual:", y_test[x])

x_prac = ([[0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 2]])
# x_prac = np.array(x_prac).reshape(len(x_prac), 1)
prac = list(pipe.predict(x_prac))
print(int(prac[0]))
print(acc)

#     if accuracy < acc:
#         accuracy = acc
#         index = i
# print(accuracy, index)
# accuracy = 0.3888, index = 88

# linear = linear_model.LinearRegression()
# linear.fit(x_train, y_train)
# acc = linear.score(x_test, y_test)

# for i in range(len(x_test)):
#     print(x_test[i], y_test[i], '\n')


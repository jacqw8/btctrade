import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import math


data = pd.read_csv('/Users/alliewu/Desktop/Chevron/filesForStartOfDatathon/training.csv')


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

GrossSoldQuantity = le.fit_transform(list(data['GrossSoldQuantity']))

d = {'StoreNumber': storenum,'dayOfTheYear':dayOfTheYear,'3HourBucket':hrbucket, 'GrossSoldQuantity':GrossSoldQuantity,
                        'EBT Site':ebt,'Loyalty Site': loyalty,'ExtraMile Site':extra,
                        'CoBrand':cobrand, 'Alcohol':alc, 'Carwash':cwash,
                        'Food Service': food, 'City':city, 'State':state}
df = pd.DataFrame(data=d)

df1 = df[df['StoreNumber']==0]
df1_sort = sorted(list(df1['GrossSoldQuantity']))
quart1 = df1_sort[int(len(df1_sort)/4)]
quart3 = df1_sort[int(len(df1_sort)*(0.75))]
interquart = quart3 - quart1
outliers_removed = []
for i in df1_sort:
    if i > (quart1-1.5*interquart) and  i < quart3+1.5*interquart:
        outliers_removed.append(i)
df1 = df1[df1['GrossSoldQuantity'].isin(outliers_removed)]


df2 = df[df['StoreNumber']==1]
df2_sort = sorted(list(df2['GrossSoldQuantity']))
quart1 = df2_sort[int(len(df2_sort)/4)]
quart3 = df2_sort[int(len(df2_sort)*(0.75))]
interquart = quart3 - quart1
outliers_removed = []
for i in df2_sort:
    if i > (quart1-1.5*interquart) and  i < quart3+1.5*interquart:
        outliers_removed.append(i)
df2 = df2[df2['GrossSoldQuantity'].isin(outliers_removed)]

df3 = df[df['StoreNumber']==2]
df3_sort = sorted(list(df3['GrossSoldQuantity']))
quart1 = df3_sort[int(len(df3_sort)/4)]
quart3 = df3_sort[int(len(df3_sort)*(0.75))]
interquart = quart3 - quart1
outliers_removed = []
for i in df3_sort:
    if i > (quart1-1.5*interquart) and  i < quart3+1.5*interquart:
        outliers_removed.append(i)
df3 = df3[df3['GrossSoldQuantity'].isin(outliers_removed)]

df4 = df[df['StoreNumber']==3]
df4_sort = sorted(list(df4['GrossSoldQuantity']))
quart1 = df4_sort[int(len(df4_sort)/4)]
quart3 = df4_sort[int(len(df4_sort)*(0.75))]
interquart = quart3 - quart1
outliers_removed = []
for i in df4_sort:
    if i > (quart1-1.5*interquart) and  i < quart3+1.5*interquart:
        outliers_removed.append(i)
df4 = df4[df4['GrossSoldQuantity'].isin(outliers_removed)]

frames = [df1, df2, df3, df4]
result = pd.concat(frames)

storenum = result['StoreNumber']
dayOfTheYear = result['dayOfTheYear']
hrbucket = result['3HourBucket']
# hrbucket = le.fit_transform(list(data['3HourBucket']))
ebt = result['EBT Site']
loyalty = result['Loyalty Site']
extra = result['ExtraMile Site']
cobrand = result['CoBrand']
alc = result['Alcohol']
cwash = result['Carwash']
food = result['Food Service']
city = result['City']
state = result['State']
GrossSoldQuantity = result['GrossSoldQuantity']

x = list(zip(storenum, dayOfTheYear, hrbucket, ebt, loyalty, extra, cobrand, alc, cwash, food, city)) #features
y = list(GrossSoldQuantity) #labels

# accuracy = 0
# index = 0
# for i in range(100):
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=46)

pipe = make_pipeline(StandardScaler(), linear_model.LinearRegression())
pipe.fit(x_train, y_train)
acc = pipe.score(x_test, y_test)

#     if accuracy < acc:
#         accuracy = acc
#         index = i
# print(accuracy, index)

predicted = pipe.predict(x_test)
for x in range(len(predicted)):
    print("Data:", x_test[x], "Predicted:", int(predicted[x]), "Actual:", y_test[x])
print(acc)
# #
# # x_prac = ([[0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 2]])
# # # x_prac = np.array(x_prac).reshape(len(x_prac), 1)
# # prac = list(pipe.predict(x_prac))
# # print(int(prac[0]))
# # print(acc)

# accuracy = 0.3888, index = 88

# linear = linear_model.LinearRegression()
# linear.fit(x_train, y_train)
# acc = linear.score(x_test, y_test)

# for i in range(len(x_test)):
#     print(x_test[i], y_test[i], '\n')


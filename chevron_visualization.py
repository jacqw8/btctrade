import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('~/Desktop/Chevron_2021_Datathon_Challenge/filesForStartOfDatathon/training.csv')

buckets = data['3HourBucket']
hotdogs = data['GrossSoldQuantity']
days = []
all_days = []

for items in data['dayOfTheYear']:
    days.append(items)
    all_days.append(items)

gross_sales = []
for items in hotdogs:
    gross_sales.append(items)

bucket_number = []
for items in buckets:
    bucket_number.append(items)

for i in range(len(days)):
    if days[i]%7 == 0:
        days[i] = 7
    else:
        days[i] = days[i]%7

Days_of_Week = [None,'Sun','Mon','Tues','Wed','Thurs','Fri','Sat']

# Figure 1
plt.bar(all_days,gross_sales, color = "orange")
plt.xlabel('Day of Year')
plt.ylabel('Hotdogs Sold')

# Figure 2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sns.set(style = "whitegrid")
ax.scatter(days, bucket_number, gross_sales, color = "orange")
ax.set_title('Hotdogs Sold Over Time')
ax.set_xticklabels(Days_of_Week)
ax.set_xlabel("Day of the Week")
ax.set_ylabel('Bucket Number')
ax.set_zlabel('Gross Sales')
plt.show()

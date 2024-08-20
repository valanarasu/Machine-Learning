import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("D:\ML\Regression\Machine-Learning\dataweather.csv")

d = {'Sunny': 0, 'Rainy': 1, 'Cloudy': 2}
df['Weather'] = df['Weather']=df['Weather'].map(d)
d = {'Weak': 0, 'Strong': 1}
df['Wind'] = df['Wind'].map(d)
d = {'YES': 1, 'NO': 0}
df['PlayOutSide'] = df['PlayOutSide'].map(d)

features = ['Weather', 'Wind']

X = df[features]
y = df['PlayOutSide']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# tree.plot_tree(dtree, feature_names=features)
print(dtree.predict([[1,1]]))
# print(dtree.predict([[40, 10, 6, 1]]))

print("[1] means 'YES'")
print("[0] means 'NO'")
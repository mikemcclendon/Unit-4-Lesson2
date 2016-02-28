import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import sklearn.metrics as skm



# data location: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
bodydata = pd.read_csv('UCI HAR Dataset/train/X_train.txt', header=None, delim_whitespace=True)
cols = pd.read_csv('UCI HAR Dataset/features.txt', header=None, delim_whitespace=True)
activities = pd.read_csv('UCI HAR Dataset/train/y_train.txt', header=None, delim_whitespace=True)
subjects = pd.read_csv('UCI HAR Dataset/train/subject_train.txt', header=None, delim_whitespace=True)

#cleaning column headers. 
labellist = []
npplist = []
nplist = []
ndlist = []
nhplist = []
nclist = []
nlclist = []
nbodylist = []
sblist = []

for row in cols.ix[:,1]:
	labellist.append(row)
for item in labellist:
	npplist.append(item.replace('()', ''))
for item in npplist:
	nplist.append(item.replace(')', ''))	
for item in nplist:
	ndlist.append(item.replace('-', ''))	
for item in ndlist:
	nclist.append(item.replace(',', ''))	
for item in nclist:
	nlclist.append(item.replace('(', ''))	
for item in nlclist:
	if item.find('Body') == True:
		nbodylist.append(item.replace('Body', ''))
	else:
		nbodylist.append(item)

#bringing cleaned features over as headers		
df = pd.DataFrame(data=bodydata.values, columns=nbodylist)

# bringing over the activities and making them categorical depending on walking, 1 if yes
# adding subjects
activdf = pd.DataFrame(data=activities.values, columns=['Activities'])
activdf['CatActivity'] = map(lambda x: 1 if x < 3 else 0, activdf['Activities'])
df['Subjects'] = subjects

# dropping band data
droplist = []
for column in df.columns:
	if column.find("bands") > 0:
		droplist.append(column)	#how do I do this without a list?
df = df.drop(droplist, 1)

#organizing data into train, test, validation
X = df
y = activdf['CatActivity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
val_X, val_y = X_test[435:], y_test[435:]
test_X, test_y = X_test[-435:], y_test[-435:]

#fitting the forest
forest = RandomForestClassifier(n_estimators = 100, oob_score=True)
forest = forest.fit(X_train,y_train) 

#printing out the top ten
fi = enumerate(forest.feature_importances_)
cols = df.columns
count = 0
print "Top Estimators Are:"
for (i,value) in fi:
	if count < 10 and value > .019:
		print (value,cols[i]) 
		count += 1

#printing out additional tests
test_pred = forest.predict(test_X)
print("Mean accuracy score for validaation set = %f" %(forest.score(val_X, val_y)))
print("Mean accuracy score for test set = %f" %(forest.score(test_X, test_y)))
print("Model Accuracy = %f" %(skm.accuracy_score(test_y,test_pred)))
print("Model Precision = %f" %(skm.precision_score(test_y,test_pred)))	
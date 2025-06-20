import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("train.csv", delimiter=',')

# Identify null values
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data)

# Preprocessing
df = df.drop(['employee_id', 'region', 'recruitment_channel'], axis=1)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['education'] = labelencoder.fit_transform(df['education'].astype(str))
df['department'] = labelencoder.fit_transform(df['department'].astype(str))
df['gender'] = pd.get_dummies(df['gender'])

df = df.replace(np.nan, df.median())

X = df.drop('is_promoted', axis=1)
y = df['is_promoted']

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# SVM with polynomial kernel
svm_poly = SVC(kernel='poly')
svm_poly.fit(X_train, y_train)
predictions_poly = svm_poly.predict(X_test)
print("SVM Poly Accuracy:", accuracy_score(y_test, predictions_poly))
print(classification_report(y_test, predictions_poly))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predictions_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, predictions_rf))
print(classification_report(y_test, predictions_rf))

# NearMiss undersampling
from imblearn.under_sampling import NearMiss
nm = NearMiss()
x_nm, y_nm = nm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(x_nm, y_nm, test_size=0.3, random_state=1)
rf.fit(X_train, y_train)
print("Random Forest (NearMiss) Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

# ROC Curve
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from numpy import sqrt, argmax

yhat = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, yhat)
gmeans = sqrt(tpr * (1-fpr))
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
pyplot.plot([0,1], [0,1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()
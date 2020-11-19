# Import necessary libraries
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Import dataset
cancer = datasets.load_breast_cancer()

# Format datasets
x = cancer.data
y = cancer.target

# Create test and training sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# Define what the values mean
classes = ['malignant', 'benign']

#Create the model
# kernel = model type (linear, rbf(default), poly, sigmoid, precomputed, callable)
# C = penalty parameter ()
clf = svm.SVC(kernel = "linear", C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print (acc)

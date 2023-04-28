import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder





data = pd.read_csv('airlines_delay.csv')

#for col in data.columns:
    #print(col, ':', len(data[col].unique()))
le = LabelEncoder()
data['Airline'] = le.fit_transform(data['Airline'])
data['AirportFrom'] = le.fit_transform(data['AirportFrom'])
data['AirportTo'] = le.fit_transform(data['AirportTo'])



delete_col = 'Flight'
target_col = 'Class'
y = data[target_col]
z = data[delete_col]

x = data.drop(target_col, axis=1)
x = x.drop(delete_col, axis=1)


X, X_, Y, Y_ = train_test_split(x, y, test_size=0.3, random_state=42)
X=X.to_numpy()
Y=Y.to_numpy()



kf = KFold(n_splits=10)

# Initialize empty lists to store the F1-score and accuracy for each fold
f1_scores = []
accuracies = []

f1_scores2 = []
accuracies2 = []

# Loop over each fold
for train_index, test_index in kf.split(X):
    print('Fold')
    # Split the data into training and test sets for this fold
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = Y[train_index], Y[test_index]

    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), activation='tanh')
    # Set the output activation function to "softmax"
    clf.out_activation_ = 'softmax'
    # Train the classifier on the training set
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)

    clf2 = MLPClassifier(solver='sgd', hidden_layer_sizes=(100, 50), activation='tanh')
    # Set the output activation function to "softmax"
    clf2.out_activation_ = 'softmax'
    # Train the classifier on the training set
    clf2.fit(X_train, y_train)

    y_pred2 = clf2.predict(X_val)

    # Calculate the F1-score and accuracy for this fold
    f1_scores.append(f1_score(y_val, y_pred, average='macro'))
    accuracies.append(accuracy_score(y_val, y_pred))

    f1_scores2.append(f1_score(y_val, y_pred2, average='macro'))
    accuracies2.append(accuracy_score(y_val, y_pred2))

# Print the F1-score and accuracy for each fold
print("a-one hidden layer with 100 hidden neurons")
for i in range(len(f1_scores)):
    print(f"Fold {i+1}: F1-score = {f1_scores[i]:.4f}, Accuracy = {accuracies[i]:.4f}")
print("The average F1-score is:",sum(f1_scores)/len(f1_scores) ,"and the average Accuracy is:", sum(accuracies)/len(accuracies))

print("b-two hidden layers with 100,50 hidden neurons")
for i in range(len(f1_scores2)):
    print(f"Fold {i+1}: F1-score = {f1_scores2[i]:.4f}, Accuracy = {accuracies2[i]:.4f}")
print("The average F1-score is:",sum(f1_scores2)/len(f1_scores2) ,"and the average Accuracy is:", sum(accuracies2)/len(accuracies2))

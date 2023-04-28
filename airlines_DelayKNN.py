import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import KFold
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



#print(x.head())
#print(y.head())
#print(x.shape)
#print(y.shape)

X, X_, Y, Y_ = train_test_split(x, y, test_size=0.3, random_state=42)
X=X.to_numpy()
Y=Y.to_numpy()
def custom_distance(x, y):
    # Compute Hamming distance for discrete features
    hamming_dist = np.sum(x[:-1] != y[:-1])
    return hamming_dist

for k in [1,3,5,10]:
    # Create a KNeighborsClassifier object with k=3 and the custom metric
    knn = KNeighborsClassifier(n_neighbors=k, metric=custom_distance)

    kf = KFold(n_splits=10)

    # Initialize empty lists to store the F1-score and accuracy for each fold
    f1_scores = []
    accuracies = []
    # Loop over each fold
    for train_index, test_index in kf.split(X):
        # Split the data into training and test sets for this fold
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = Y[train_index], Y[test_index]

        # Fit the KNN classifier on the training data
        knn.fit(X_train, y_train)

        # Predict the test labels using the trained KNN classifier
        y_pred = knn.predict(X_val)

        # Calculate the F1-score and accuracy for this fold
        f1_scores.append(f1_score(y_val, y_pred, average='macro'))
        accuracies.append(accuracy_score(y_val, y_pred))

    # Print the F1-score and accuracy for each fold
    for i in range(len(f1_scores)):
        print(f"Fold {i+1}: F1-score = {f1_scores[i]:.4f}, Accuracy = {accuracies[i]:.4f}")
    print("For",k,"neighbors the average F1-score is:",sum(f1_scores)/len(f1_scores) ,"and the average Accuracy is:", sum(accuracies)/len(accuracies))





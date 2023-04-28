
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import KFold


data = pd.read_csv('train.csv')

features_to_move = ['blue', 'clock_speed', 'dual_sim', 'fc', 'four_g','int_memory', 'm_dep', 'n_cores', 'pc', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']
features_to_stay = [col for col in data.columns if col not in features_to_move]
data = data[features_to_move + features_to_stay]


target_col = 'price_range'
y = data[target_col]

x = data.drop(target_col, axis=1)

X, X_, Y, Y_ = train_test_split(x, y, test_size=0.3, random_state=42)
X = X.to_numpy()
Y = Y.to_numpy()


def custom_distance(x, y):
    # Compute Hamming distance for discrete features
    hamming_dist = np.sum(x[:len(features_to_move)-1] != y[:len(features_to_move)-1])
    # Compute Euclidean distance for continuous features
    euclidean_dist = np.linalg.norm(x[len(features_to_move)-1:] - y[len(features_to_move)-1:])
    # Return the combined distance
    return hamming_dist + euclidean_dist


for k in [1, 3, 5, 10]:
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
    print("For", k, "neighbors the average F1-score is:", sum(f1_scores)/len(f1_scores),
          "and the average Accuracy is:", sum(accuracies)/len(accuracies))





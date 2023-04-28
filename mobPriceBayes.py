import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv('train.csv')
#for col in data.columns:
    #print(col, ':', len(data[col].unique()))
features_to_move = ['blue', 'clock_speed', 'dual_sim', 'fc', 'four_g','int_memory', 'm_dep', 'n_cores', 'pc', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']
features_to_stay = [col for col in data.columns if col not in features_to_move]
data = data[features_to_move + features_to_stay]

#make indexes for each feature
discrete_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
continuous_features = [15, 16, 17, 18, 19]

target_col = 'price_range'
y = data[target_col]

x = data.drop(target_col, axis=1)


X, X_, Y, Y_ = train_test_split(x, y, test_size=0.3, random_state=42)
X=X.to_numpy()
Y=Y.to_numpy()



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
    multi_clf = MultinomialNB()
    multi_clf.fit(X_train[:, discrete_features], y_train)

    gauss_clf = GaussianNB()
    gauss_clf.fit(X_train[:, continuous_features], y_train)

    multi_probs = multi_clf.predict_proba(X_val[:, discrete_features])
    gauss_probs = gauss_clf.predict_proba(X_val[:, continuous_features])

    # Combine the probabilities using the product rule
    probs = np.multiply(multi_probs, gauss_probs)

    # Predict the class with the highest probability
    y_pred = np.argmax(probs, axis=1)



    # Calculate the F1-score and accuracy for this fold
    f1_scores.append(f1_score(y_val, y_pred, average='macro'))
    accuracies.append(accuracy_score(y_val, y_pred))

# Print the F1-score and accuracy for each fold
for i in range(len(f1_scores)):
    print(f"Fold {i+1}: F1-score = {f1_scores[i]:.4f}, Accuracy = {accuracies[i]:.4f}")
print("The average F1-score is:",sum(f1_scores)/len(f1_scores) ,"and the average Accuracy is:", sum(accuracies)/len(accuracies))

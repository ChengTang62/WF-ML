import constants as ct
import argparse
import joblib
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from math import sqrt

def cross_validate(X, y, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler = preprocessing.MinMaxScaler((-1, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = RandomForestClassifier(n_estimators=100, random_state=42) 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return accuracies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10 fold cross-validation for cumul attack")
    parser.add_argument("feature_dic", help="Extracted features file path.")
    args = parser.parse_args()
    dic = np.load(args.feature_dic, allow_pickle=True).item()
    X = np.array(dic['feature'])
    y = np.array(dic['label'])
    acc_scores = cross_validate(X, y, 10)
    mean_accuracy = np.mean(acc_scores)
    std_dev = np.std(acc_scores)
    print('cumul')
    print('10-fold Cross Validation Accuracy Scores:', acc_scores)
    print('Mean Accuracy:', mean_accuracy)
    print('Standard Deviation of Accuracy:', std_dev)
    print(acc_scores)
    z_value = 1.96
    margin_of_error = z_value * (std_dev / sqrt(10))
    lower_bound = mean_accuracy - margin_of_error
    upper_bound = mean_accuracy + margin_of_error
    confidence_interval = (lower_bound, upper_bound)
    print("95% Confidence Interval: {:.4f} to {:.4f}".format(*confidence_interval))    
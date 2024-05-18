import joblib
import pickle
import const as ct
import logging
import argparse
import configparser
import numpy as np
import multiprocessing
from math import sqrt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

FEATURE_NAMES = [
    "Interarrival Time Max In", "Interarrival Time Max Out", "Interarrival Time Max Total",
    "Interarrival Time Mean In", "Interarrival Time Mean Out", "Interarrival Time Mean Total",
    "Interarrival Time SD In", "Interarrival Time SD Out", "Interarrival Time SD Total",
    "Interarrival Time 75th percentile In", "Interarrival Time 75th percentile Out", "Interarrival Time 75th percentile Total",
    "Time Percentile 25 In", "Time Percentile 50 In", "Time Percentile 75 In", "Time Percentile 100 In",
    "Time Percentile 25 Out", "Time Percentile 50 Out", "Time Percentile 75 Out", "Time Percentile 100 Out",
    "Time Percentile 25 Total", "Time Percentile 50 Total", "Time Percentile 75 Total", "Time Percentile 100 Total",
    "Number of Inbound Packets", "Number of Outbound Packets", "Total Number of Packets",
    "First 30 Packets Inbound", "First 30 Packets Outbound",
    "Last 30 Packets Inbound", "Last 30 Packets Outbound",
    "Packet Concentration Std Dev", "Packet Concentration Average",
    "Packets Per Second Average", "Packets Per Second Std Dev", "Packets Per Second Median", "Packets Per Second Min", "Packets Per Second Max",
    "Average Packet Ordering Inbound", "Average Packet Ordering Outbound",
    "Std Dev Packet Ordering Inbound", "Std Dev Packet Ordering Outbound",
    "Percentage Inbound Packets", "Percentage Outbound Packets",
]+ [
    "Packet Concentration Array {}".format(i) for i in range(1, 72)
] + [
    "Packets Per Second Array {}".format(i) for i in range(1, 21)
]


def kfingerprinting(X, y):
    model = RandomForestClassifier(n_jobs=-1, n_estimators=1000, oob_score=True)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1123)
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    mean_accuracy = np.mean(acc_scores)
    std_dev = np.std(acc_scores)
    print('kfingerprinting')
    print('10-fold Cross Validation Accuracy Scores:', acc_scores)
    print('Mean Accuracy:', mean_accuracy)
    print('Standard Deviation of Accuracy:', std_dev)
    print(acc_scores)
    z_value = 1.96  # For a 95% confidence interval
    margin_of_error = z_value * (std_dev / sqrt(10))
    lower_bound = mean_accuracy - margin_of_error
    upper_bound = mean_accuracy + margin_of_error
    confidence_interval = (lower_bound, upper_bound)
    print("95% Confidence Interval: {:.4f} to {:.4f}".format(*confidence_interval))    
    model.fit(X, y)
    joblib.dump(model, 'ranpad2_0610_2057_norm.pkl')
    feature_importances = model.feature_importances_
    return model, feature_importances

if __name__ == '__main__':

    dic = np.load("/Users/ct/Library/Mobile Documents/com~apple~CloudDocs/cybersecurity_robotics/WebsiteFingerprinting/attacks/kfingerprinting/results/torque_data.npy", allow_pickle=True).item()
    
    X = np.array(dic['feature'])
    Y = np.array(dic['label'])
    y = np.array([label[0] for label in Y])
    model, importances = kfingerprinting(X, y)
    importances = np.array(importances)
    sorted_indices = np.argsort(importances)[::-1]
    for index in sorted_indices:
        print(sorted_indices)
        print(FEATURE_NAMES[index], importances[index])
    indices = np.argsort(importances)[::-1]

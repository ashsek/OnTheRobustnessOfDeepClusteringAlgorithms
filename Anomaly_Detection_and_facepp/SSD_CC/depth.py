#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.stats import chi2
import sys


try:
    dataset = str(sys.argv[1])
    threshold_quantile = float(sys.argv[2])
    
except:
    dataset = 'iris.csv'
    threshold_quantile = 0.98


df = pd.read_csv(dataset)
print("Dataset Loaded: " + dataset)
print(df.head())

df = df.dropna()
df = df.to_numpy()


num_samples = df.shape[0]
num_features = df.shape[1]


cov  = np.cov(df , rowvar=False)
cov = np.linalg.matrix_power(cov, -1)
mean = np.mean(df , axis=0)


dists = []
for i, val in enumerate(df):
    a = val
    b = mean
    dist = (a-b).T.dot(cov).dot(a-b)
    dists.append(dist)
dists = np.array(dists)

threshold = chi2.ppf(threshold_quantile, num_features)

outliers, indxs = [], []
for idx, dist in enumerate(dists):
    if dist > threshold:
        indxs.append(idx)
        outliers.append(df[idx, :])

print('\nOutliers found at following indices:')
print(indxs)
print('\nData Samples (Outliers) corresponding to these indices:')
print(outliers)
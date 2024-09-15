import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plotting import *
import hmmlearn.hmm as hmm
import hmmlearn.vhmm as vhmm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import logging
import warnings

logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

N_OBS           = 3         # Number of observations
SAMPLE_SPACE    = 25       # Number of samples in the sample space
MAX_ATTEMPTS    = 5        # Maximum number of attempts to fit the model


# K-Means Clustering to initialize the HMM
def kmeans_gmm_hmm(train_data, n_states, company):
    kmeans = KMeans(n_clusters=n_states)
    kmeans.fit(train_data[company][["fracChange","fracHigh","fracLow"]])
    means = kmeans.cluster_centers_
    labels = kmeans.labels_

    means = np.zeros((n_states, N_OBS))
    covars = np.zeros((n_states, N_OBS))
    weights = np.zeros((n_states))
    for state in range(n_states):
        cluster_data = train_data[company][["fracChange","fracHigh","fracLow"]][labels == state]
        if len(cluster_data) > 0:
            means[state] = np.mean(cluster_data, axis=0)
            covars[state] = np.var(cluster_data, axis=0)
            weights[state] = len(cluster_data) / len(train_data[company])

    weights /= weights.sum()  # Normalize weights correctly

    # Gaussian Model Hidden Markov Model Initialization
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", init_params='', algorithm="map")
    model.startprob_ = np.full(n_states, 1/n_states)  # Initial state probabilities are equal for all states, ergodic model
    model.transmat_ = np.full((n_states, n_states), 1/n_states)  # Transition probabilities are equal for all states, ergodic model
    model.means_ = means
    model.covars_ = covars
    model.weights_ = weights
    model.n_features = N_OBS

    return model

def VI_HMM(train_data, n_states, company, latency):

    model = vhmm.VariationalGaussianHMM(n_components=n_states, covariance_type="full", algorithm="viterbi", n_iter=10)
    
    train = train_data[company][["fracChange", "fracHigh", "fracLow"]]
    new_train = []
    for i in range(len(train) - latency):
        window = train.iloc[i:i + latency + 1].values
        if len(window) == latency + 1:
            new_train.append(window)

    new_train = np.array(new_train)

    X = np.concatenate(new_train)

    lengths = [len(x) for x in new_train]

    attempts = 0
    success = False
    
    while not success and attempts < MAX_ATTEMPTS:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                model.fit(X, lengths=lengths)
            success = True
        except RuntimeWarning:
            attempts += 1
            print(f"RuntimeWarning encountered. Retrying... (Attempt {attempts}/{MAX_ATTEMPTS})")
    
    if not success:
        raise RuntimeError(f"Failed to fit model after {MAX_ATTEMPTS} attempts due to RuntimeWarning.")

    return model


def EM_HMM(train_data, n_states, company, latency):

    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", algorithm="viterbi", n_iter=10)

    train = train_data[company][["fracChange", "fracHigh", "fracLow"]]
    new_train = []
    for i in range(len(train) - latency):
        window = train.iloc[i:i + latency + 1].values
        if len(window) == latency + 1:
            new_train.append(window)

    new_train = np.array(new_train)

    X = np.concatenate(new_train)

    lengths = [len(x) for x in new_train]

    attempts = 0
    success = False
    
    while not success and attempts < MAX_ATTEMPTS:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                model.fit(X, lengths=lengths)
            success = True
        except RuntimeWarning:
            attempts += 1
            print(f"RuntimeWarning encountered. Retrying... (Attempt {attempts}/{MAX_ATTEMPTS})")
    
    if not success:
        raise RuntimeError(f"Failed to fit model after {MAX_ATTEMPTS} attempts due to RuntimeWarning.")

    return model
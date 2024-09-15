import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plotting import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from data.extractdata import StockDataExtractor
import requests
import logging
import collections
from models import VI_HMM, EM_HMM, kmeans_gmm_hmm


logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

N_OBS           = 3        # Number of observations
SAMPLE_SPACE    = 75       # Number of samples in the sample space

def find_best_model(train_data, n_states, company, latency):
    best_scores = collections.defaultdict(dict)
    best_models = collections.defaultdict(dict)
    
    for state in range(1, n_states + 1):
        # Train VI HMM
        vi = VI_HMM(train_data, state, company, latency)
        lb = vi.monitor_.history[-1] if hasattr(vi, 'monitor_') else None
        print(f"VI HMM with {state} states for {company} has log likelihood of {lb}")
        best_models["VI"][state] = vi
        best_scores["VI"][state] = lb if lb is not None else float('-inf')

        # Train EM HMM
        em = EM_HMM(train_data, state, company, latency)
        ll = em.monitor_.history[-1] if hasattr(em, 'monitor_') else None
        print(f"EM HMM with {state} states for {company} has log likelihood of {ll}")
        best_models["EM"][state] = em
        best_scores["EM"][state] = ll if ll is not None else float('-inf')

        # Train K-Means HMM
        kmeans = kmeans_gmm_hmm(train_data, state, company)
        bic_score = bic(kmeans, train_data[company][["fracChange", "fracHigh", "fracLow"]])
        print(f"K-Means HMM with {state} states for {company} has BIC score of {bic_score}")
        best_models["KMeans"][state] = kmeans
        best_scores["KMeans"][state] = bic_score

    best_vi_state, best_vi_score = max(best_scores["VI"].items(), key=lambda x: x[1])
    best_em_state, best_em_score = max(best_scores["EM"].items(), key=lambda x: x[1])
    best_kmeans_state, best_kmeans_score = min(best_scores["KMeans"].items(), key=lambda x: x[1])

    # print(f"Best VI Model for {company}: {best_vi_state} states with score {best_vi_score}")
    # print(f"Best EM Model for {company}: {best_em_state} states with score {best_em_score}")
    # print(f"Best K-Means Model for {company}: {best_kmeans_state} states with score {best_kmeans_score}")

    return best_models["VI"][best_vi_state], best_models["EM"][best_em_state], best_models["KMeans"][best_kmeans_state]

def stock_price_prediction(latency, company, model):
    predicted_closeprices = []
    actual_closeprices = []

    for i in tqdm(range(test_data_wo_date.shape[0]-1)):

        sample_space_fracChange = np.linspace(np.min(train_data_wo_date[company]["fracChange"]), np.max(train_data_wo_date[company]["fracChange"]), SAMPLE_SPACE)
        sample_space_fracHigh = np.linspace(0, np.max(train_data_wo_date[company]["fracHigh"]), int(SAMPLE_SPACE / 5))
        sample_space_fracLow = np.linspace(0, np.max(train_data_wo_date[company]["fracLow"]), int(SAMPLE_SPACE / 5))
        possible_outcome = np.array(list(itertools.product(sample_space_fracChange, sample_space_fracHigh, sample_space_fracLow)))

        outcome_score = np.zeros(len(possible_outcome))

        if i < latency:
            test = pd.concat([train_data_wo_date[company][-latency + i:], test_data_wo_date[company][:i]])
        else:
            test = test_data_wo_date[company][i - latency:i]

        if test.empty or len(possible_outcome) == 0:
            continue  # Skip if there are no valid test data

        for j in range(len(possible_outcome)):
            outcome = np.vstack([test[["fracChange", "fracHigh", "fracLow"]], possible_outcome[j]])
            outcome_score[j] = model.score(outcome)

        best_outcome = possible_outcome[np.argmax(outcome_score)]
        predicted_closeprice = test_data[company]["Open"].iloc[i] * (1 + best_outcome[0])
        actual_closeprice = test_data[company]["Open"].iloc[i + 1]

        predicted_closeprices.append(predicted_closeprice)
        actual_closeprices.append(actual_closeprice)

    return predicted_closeprices, actual_closeprices


def mape(predicted_closeprices, actual_closeprices):

    mape_error = np.mean(np.abs(np.array(predicted_closeprices) - np.array(actual_closeprices)) / np.abs(np.array(actual_closeprices)))*100
    mape_error_std = np.std(np.abs(np.array(predicted_closeprices) - np.array(actual_closeprices)) / np.abs(np.array(actual_closeprices)))*100
    
    return mape_error, mape_error_std

def bic(model, data):
    return -2*model.score(data) + model.n_components*np.log(data.shape[0])

# Check internet connectivity
def check_internet():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

############################################################################################################
# Main code
############################################################################################################

# Hyperparameters
n_states = 7 # Number of states in the HMM
latencies = [2, 4, 7, 14] # Number of days to predict in the future

# Download and preprocess data
dataset = StockDataExtractor()

if check_internet():
    # companies = ["AAPL" , "MSFT", "NVDA", "GOOGL", "META"]
    companies = ["AAPL", "NVDA", "MSFT"]
    dataset.download_data(companies)
    dataset.preprocess_data()
else:
    # Load data from local files
    companies = ["AAPL", "MSFT", "NVDA"]
    dataset.load_data_preprocessed()

# Divide dataset into training and testing divided for data
train_data = dataset.data2[(dataset.data2.index < "2024-01-01") & (dataset.data2.index > "2020-01-01")]
test_data = dataset.data2[(dataset.data2.index >= "2024-01-01")] 
train_data_wo_date = train_data.reset_index(drop=True)
test_data_wo_date = test_data.reset_index(drop=True)

for company in tqdm(companies):
    mape_errors_vi = []
    mape_errors_stds_vi = []
    mape_errors_em = []
    mape_errors_stds_em = []
    mape_errors_km = []
    mape_errors_std_km = []

    for latency in tqdm(latencies):

        print(f" HMM with for {company} with latency of {latency} days")
        model_vi, model_em, model_km = find_best_model(train_data_wo_date, n_states, company, latency)
        # visualize_states(model_vi.n_components, train_data, train_data_wo_date, company, model_vi, method="VI", latency=latency)
        # visualize_states(model_em.n_components, train_data, train_data_wo_date, company, model_em, method="EM", latency=latency)
        # visualize_states(model_km.n_components, train_data, train_data_wo_date, company, model_km, method="KMeans", latency=latency)

        predicted_closeprices_vi, actual_closeprices_vi = stock_price_prediction(latency, company, model_vi)
        plot_predicted_vs_actual_close_prices(test_data, company, predicted_closeprices_vi, actual_closeprices_vi, latency, method="VI")

        predicted_closeprices_em, actual_closeprices_em = stock_price_prediction(latency, company, model_em)
        plot_predicted_vs_actual_close_prices(test_data, company, predicted_closeprices_em, actual_closeprices_em, latency, method="EM")

        predicted_closeprices_km, actual_closeprices_km = stock_price_prediction(latency, company, model_km)
        plot_predicted_vs_actual_close_prices(test_data, company, predicted_closeprices_km, actual_closeprices_km, latency, method="KMeans")

        mape_error_vi, mape_error_std_vi = mape(predicted_closeprices_vi, actual_closeprices_vi)
        mape_error_em, mape_error_std_em = mape(predicted_closeprices_em, actual_closeprices_em)
        mape_error_km, mape_error_std_km = mape(predicted_closeprices_km, actual_closeprices_km)

        print(f"MAPE Error for {company} with latency of {latency} days (VI): {mape_error_vi}")
        print(f"MAPE Error Standard Deviation for {company} with latency of {latency} days (VI): {mape_error_std_vi}")
        print(f"MAPE Error for {company} with latency of {latency} days (EM): {mape_error_em}")
        print(f"MAPE Error Standard Deviation for {company} with latency of {latency} days (EM): {mape_error_std_em}")
        print(f"MAPE Error for {company} with latency of {latency} days (KMeans): {mape_error_km}")
        print(f"MAPE Error Standard Deviation for {company} with latency of {latency} days (KMeans): {mape_error_std_km}")

        mape_errors_vi.append(mape_error_vi)
        mape_errors_stds_vi.append(mape_error_std_vi)
        mape_errors_em.append(mape_error_em)
        mape_errors_stds_em.append(mape_error_std_em)
        mape_errors_km.append(mape_error_km)
        mape_errors_std_km.append(mape_error_std_km)

    plot_mape_errors(latencies, mape_errors_vi, mape_errors_stds_vi, company, method="VI")
    plot_mape_errors(latencies, mape_errors_em, mape_errors_stds_em, company, method="EM")
    plot_mape_errors(latencies, mape_errors_km, mape_errors_std_km, company, method="KMeans")

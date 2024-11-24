import numpy as np
from scipy.fftpack import fft, ifft
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
from scipy.spatial.distance import euclidean
from data_processing import chunk_time_series, augment_data
from dtaidistance import dtw
from fastdtw import fastdtw
"""
Feature extraction fonctions

Functions:
    extract_features_model2(docs)
    find_cluster_centers(train_docs)
    calculate_means_and_interval_model1(docs)
    compute_cross_distance_fft(transmission)
"""

#fonction to extract features from normalized transmission data
#in this we use parallelized feature extraction to improve computation time
#decide when to use gram type as a feature

def extract_features_model2(docs, use_gram_type_as_feature=False, use_shape_type_as_feature=False):
    def process_doc(doc):
        transmission = doc['transmission_normalized']
        mean_transmission = np.mean(transmission)
        std_transmission = np.std(transmission)
        max_transmission = np.max(transmission)
        min_transmission = np.min(transmission)
        skewness = skew(transmission)
        kurt = kurtosis(transmission)
        cross_distance = compute_cross_distance_fft(transmission)
        feature_vector = [skewness, std_transmission,kurt, cross_distance, mean_transmission] #i removed min transmission et max, mean
        if use_gram_type_as_feature:
            feature_vector.append(doc['label'])
        if use_shape_type_as_feature:
            feature_vector.append(doc['shape'])
        return feature_vector, doc['label'], doc['bacteria'], doc['shape']
    results = Parallel(n_jobs=-1)(delayed(process_doc)(doc) for doc in docs)
    features, gram_labels, bacteria_labels, shape_labels = zip(*results)
    return np.array(features), np.array(gram_labels), np.array(bacteria_labels), np.array(shape_labels)





#finds the cluster centers for each family
def find_cluster_centers(train_docs):
    cluster_centers = {}
    bacteria_families = set([doc['bacteria'] for doc in train_docs])
    for family in bacteria_families:
        family_docs = [doc for doc in train_docs if doc['bacteria'] == family]
        mean_values = [np.mean(doc['transmission_normalized']) for doc in family_docs]
        std_values = [np.std(doc['transmission_normalized']) for doc in family_docs]
        center_mean = np.mean(mean_values)
        center_std = np.mean(std_values)
        cluster_centers[family] = (center_mean, center_std)
    return cluster_centers


#let's calculate the mean and this confidence interval here the doc is supposed to be normalized
def calculate_means_and_interval_model1(docs):
    gram_pos_means = []
    gram_neg_means = []
    for doc in docs:
        mean_value = np.mean(doc['transmission_normalized'])
        if doc['label'] == 1:
            gram_pos_means.append(mean_value)
        elif doc['label'] == 0:
            gram_neg_means.append(mean_value)
    interval_min = max(gram_pos_means)
    interval_max = min(gram_neg_means)
    return interval_min, interval_max


# Compute cross-distance (auto-distance), feature suggested in the thesis epfl n10236 p 120
# This measures the self-similarity of the signal over time.
# Compute cross-distance using FFT (fast Fourier transform)
# Purpose: Provides a compact representation of the signal self similarity. It captures how the signal behaves when shifted over time
# The autocorrelation helps in understanding periodicity and repeating patterns within the transmission signal. How the signal correlates with itself over different lags
# We used FFT and IFFT and not other time domain correlation because of computational efficiency
def compute_cross_distance_fft(transmission):
    T = len(transmission)
    f = fft(transmission)  #computes the FFT of the transmission signal transforming it from the time domain to the frequency domain
    power_spectrum = np.abs(f) ** 2  #shows the distribution of power into the frequency components making up the signal
    autocorrelation = ifft(power_spectrum).real  # Inverse FFT : IFFT, obtains the autocorrelation of the signal, which measures how the signal correlates with a delayed version of itself over varying delays or lags
    cross_distances = np.sqrt(np.sum((transmission - autocorrelation[:T]) ** 2))  # Euclidean distance between the original transmission signal and its autocorrelation, measures the difference between the transmission signal and the autocorrelated signal up to length T
    return np.min(cross_distances)









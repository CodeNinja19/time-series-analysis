import pandas as pd
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import numpy as np
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import extract_features

settings = ComprehensiveFCParameters()  # Creating a Settings object

data = pd.read_csv('deepface_results.csv')
SE = pd.read_csv('task3.csv')
AUs = data[['ID', 'frame', 'happiness', 'sadness', 'fear', 'disgust']]

# Get the label column and make sure the label data is Series Get the label column and make sure the label data is Series
class_SE = SE['self-esteem']
parameters = {"abs_energy": None,
              'kurtosis': None,
              'skewness': None,
              'mean': None,
              'standard_deviation': None,
              'variation_coefficient': None,
              'median': None,
              'minimum': None,
              'maximum': None,
              'mean_change': None,
              'mean_abs_change': None,
              'number_peaks': [{'n': 4}],
              'sum_values': None,
              "cid_ce": [{'normalize': True}],
              'sample_entropy': None,
              "approximate_entropy": [{"m": 5, "r": 0.1}],
              "fft_coefficient": [{"coeff": 1, "attr": 'abs'}, {"coeff": 1, "attr": 'real'}, {"coeff": 1, "attr": 'imag'}],
              'ar_coefficient': [{'coeff': 1, 'k': 5}],
              'autocorrelation': [{'lag': 5}],
              'linear_trend': [{'attr': 'rvalue'}, {'attr': 'intercept'}, {'attr': 'slope'}],  # “pvalue”, “rvalue”, “intercept”, “slope”
              }

def main():

    # Using AUs data for feature extraction ensures that AUs are in DataFrame format and contain the correct column names
    extracted_features = extract_features(AUs, default_fc_parameters=parameters, column_id="ID", column_sort="frame")
    # Imputation of data (if missing)
    impute(extracted_features)
    extracted_features.index = np.arange(0, len(extracted_features))

    # saving results
    extracted_features.to_csv("20250227.csv", index=False)
    print('Finished')

if __name__ == '__main__':
    main()
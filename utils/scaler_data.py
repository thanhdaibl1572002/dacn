import pandas as pd
def scaler_data():
    datasets = {
        "Scaler Regression Big": pd.read_csv('dacn/regression_big.csv'),
        "Scaler Classification Big": pd.read_csv('dacn/classification_big.csv'),
    }
    return datasets
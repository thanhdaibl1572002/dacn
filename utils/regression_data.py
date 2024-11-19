import pandas as pd
from sklearn.preprocessing import StandardScaler
def regression_data():
    datasets = {
        "Regression Big": pd.read_csv('dacn/regression_big.csv'),
        "Regression Medium": pd.read_csv('dacn/regression_medium.csv'),
        "Regression Small": pd.read_csv('dacn/regression_small.csv')
    }
    new = pd.read_csv('dacn/regression_new.csv')
    target = 'ArrDelay'
    scaler = StandardScaler()
    for key in datasets:
        X = datasets[key].drop(columns=[target])
        y = datasets[key][target]
        X_scaled = scaler.fit_transform(X)
        datasets[key] = pd.DataFrame(X_scaled, columns=X.columns)
        datasets[key][target] = y
    X_new_scaled = scaler.transform(new)
    new = pd.DataFrame(X_new_scaled, columns=new.columns)
    return datasets, new, target
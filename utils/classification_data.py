import pandas as pd
from sklearn.preprocessing import StandardScaler
def classification_data():
    datasets = {
        "Classification Big": pd.read_csv('dacn/classification_big.csv'),
        "Classification Medium": pd.read_csv('dacn/classification_medium.csv'),
        "Classification Small": pd.read_csv('dacn/classification_small.csv')
    }
    new = pd.read_csv('dacn/classification_new.csv')
    target = 'HasDiabetes'
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
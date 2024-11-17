import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================== UTILS =============================== #
def split_data(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=[target]).values
    Y = df[target].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test

def get_model_name(model):
    return ' '.join(re.findall(r'[A-Z][a-z]*', type(model).__name__))

def sigmoid(z): 
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def softmax(z): 
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def mse_loss(predictions, actual):
    return np.mean((predictions - actual) ** 2)

def cross_entropy_loss(predictions, actual):
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(actual * np.log(predictions) + (1 - actual) * np.log(1 - predictions))

# =============================== SCALER =============================== #
def scaler_data():
    datasets = {
        "Scaler Regression Big": pd.read_csv('dacn/regression_big.csv'),
        "Scaler Classification Big": pd.read_csv('dacn/classification_big.csv'),
    }
    return datasets

def scaler_evaluate(df, my_model, sk_model):
    start_time = time.time()
    my_model_scaled = my_model.fit_transform(df)
    print(f"My Model Scaler [0]: \n{my_model_scaled[0]}\n")
    my_time = time.time() - start_time
    start_time = time.time()
    sk_model_scaled = sk_model.fit_transform(df)
    print(f"SK Model Scaler [0]: \n{sk_model_scaled[0]}\n")
    sk_time = time.time() - start_time
    my_time_s = f"{my_time:.3f} s"
    sk_time_s = f"{sk_time:.3f} s"
    mae = f"{mean_absolute_error(my_model_scaled, sk_model_scaled):.3f}"
    rmse = f"{root_mean_squared_error(my_model_scaled, sk_model_scaled):.3f}"
    r2 = f"{r2_score(my_model_scaled, sk_model_scaled):.3f}"
    print(f"{'':<5} {'My Model':<25} {'SK Model':<25}")
    print(f"{'Time':<5} {my_time_s:<25} {sk_time_s}")
    print(f"MAE, RMSE, R2: {mae}, {rmse}, {r2}")
    return my_time, sk_time, mae, mae, rmse, r2

def scaler_evaluates(my_model, sk_model):
    datasets = scaler_data()
    my_times = []
    sk_times = []
    dataset_names = []
    model_name = get_model_name(my_model)
    for dataset_name, df in datasets.items():
        display(Markdown(f"**{model_name} - {dataset_name}**"))
        results = scaler_evaluate(df, my_model, sk_model)
        my_times.append(results[0])
        sk_times.append(results[1])
        dataset_names.append(dataset_name)
    display(Markdown(f"**{model_name} - Time Comparison**"))
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(dataset_names))
    ax.bar(x, my_times, width=0.4, label='My Model', align='center', color='skyblue')
    ax.bar([i + 0.4 for i in x], sk_times, width=0.4, label='SK Model', align='center', color='lightgreen')
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Time Comparison of My Model vs SK Model on Different Datasets")
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(dataset_names)
    ax.legend()
    plt.show()

# =============================== CLASSIFICATION =============================== #
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

def classification_evaluate(df, target, my_model, sk_model):
    X_train, X_test, Y_train, Y_test = split_data(df, target=target)
    start_time = time.time()
    my_model.fit(X_train, Y_train)
    my_Y_pred = my_model.predict(X_test)
    my_time = time.time() - start_time
    start_time = time.time()
    sk_model.fit(X_train, Y_train)
    sk_Y_pred = sk_model.predict(X_test)
    sk_time = time.time() - start_time
    my_time_s = f"{my_time:.3f} (s)"
    sk_time_s = f"{sk_time:.3f} (s)"
    my_acc_percent = f"{accuracy_score(Y_test, my_Y_pred) * 100:.3f} (%)"
    sk_acc_percent = f"{accuracy_score(Y_test, sk_Y_pred) * 100:.3f} (%)"
    my_f1_score = f"{f1_score(Y_test, my_Y_pred, average='weighted'):.3f}"
    sk_f1_score = f"{f1_score(Y_test, sk_Y_pred, average='weighted'):.3f}"
    print(f"\n{'':<5} {'My Model':<25} {'SK Model':<25}")
    print(f"{'Time':<5} {my_time_s:<25} {sk_time_s}")
    print(f"{'Acc':<5} {my_acc_percent:<25} {sk_acc_percent}")
    print(f"{'F1':<5} {my_f1_score:<25} {sk_f1_score}\n")
    cr_my = classification_report(Y_test, my_Y_pred, output_dict=True)
    cr_sk = classification_report(Y_test, sk_Y_pred, output_dict=True)
    cr_my_df = pd.DataFrame(cr_my).transpose()
    cr_sk_df = pd.DataFrame(cr_sk).transpose()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cr_my_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f', ax=axes[0], cbar=False)
    axes[0].set_title('Classification Report for My Model')
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Classes')
    sns.heatmap(cr_sk_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f', ax=axes[1], cbar=False)
    axes[1].set_title('Classification Report for SK Model')
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Classes')
    plt.tight_layout()
    plt.show()
    return my_time, sk_time

def classification_evaluates(my_model, sk_model):
    datasets, _, target = classification_data()
    my_times = []
    sk_times = []
    dataset_names = []
    model_name = get_model_name(my_model)
    for dataset_name, df in datasets.items():
        display(Markdown(f"**{model_name} - {dataset_name}**"))
        my_time, sk_time = classification_evaluate(df, target, my_model, sk_model)
        my_times.append(my_time)
        sk_times.append(sk_time)
        dataset_names.append(dataset_name)
    display(Markdown(f"**{model_name} - Time Comparison**"))
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(dataset_names))
    ax.bar(x, my_times, width=0.4, label='My Model', align='center', color='skyblue')
    ax.bar([i + 0.4 for i in x], sk_times, width=0.4, label='SK Model', align='center', color='lightgreen')
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Time Comparison of My Model vs SK Model on Different Datasets")
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(dataset_names)
    ax.legend()
    plt.show()
    
def classification_predictions(my_model):
    _, new, target = classification_data()
    predictions = my_model.predict(new)
    new_with_predictions = new.copy()
    new_with_predictions[target + " (Predicted)"] = predictions
    model_name = get_model_name(my_model)
    display(Markdown(f"**{model_name} - Predictions On New Data**"))
    return new_with_predictions
    
# =============================== REGRESSION =============================== #
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

def regression_evaluate(df, target, my_model, sk_model):
    X_train, X_test, Y_train, Y_test = split_data(df, target=target)
    start_time = time.time()
    my_model.fit(X_train, Y_train)
    my_Y_pred = my_model.predict(X_test)
    my_time = time.time() - start_time
    start_time = time.time()
    sk_model.fit(X_train, Y_train)
    sk_Y_pred = sk_model.predict(X_test)
    sk_time = time.time() - start_time
    my_time_s = f"{my_time:.3f} s"
    sk_time_s = f"{sk_time:.3f} s"
    my_mae = f"{mean_absolute_error(Y_test, my_Y_pred):.3f}"
    sk_mae = f"{mean_absolute_error(Y_test, sk_Y_pred):.3f}"
    my_rmse = f"{root_mean_squared_error(Y_test, my_Y_pred):.3f}"
    sk_rmse = f"{root_mean_squared_error(Y_test, sk_Y_pred):.3f}"
    my_r2 = f"{r2_score(Y_test, my_Y_pred):.3f}"
    sk_r2 = f"{r2_score(Y_test, sk_Y_pred):.3f}"
    print(f"\n{'':<5} {'My Model':<25} {'SK Model':<25}")
    print(f"{'Time':<5} {my_time_s:<25} {sk_time_s}")
    print(f"{'MAE':<5} {my_mae:<25} {sk_mae}")
    print(f"{'RMSE':<5} {my_rmse:<25} {sk_rmse}")
    print(f"{'R2':<5} {my_r2:<25} {sk_r2}\n")
    return my_time, sk_time, my_mae, sk_mae, my_rmse, sk_rmse, my_r2, sk_r2

def regression_evaluates(my_model, sk_model):
    datasets, _, target = regression_data()
    my_times = []
    sk_times = []
    my_maes = []
    sk_maes = []
    my_rmses = []
    sk_rmses = []
    my_r2s = []
    sk_r2s = []
    dataset_names = []
    model_name = get_model_name(my_model)
    for dataset_name, df in datasets.items():
        display(Markdown(f"**{model_name} - {dataset_name}**"))
        results = regression_evaluate(df, target, my_model, sk_model)
        my_times.append(results[0])
        sk_times.append(results[1])
        my_maes.append(results[2])
        sk_maes.append(results[3])
        my_rmses.append(results[4])
        sk_rmses.append(results[5])
        my_r2s.append(results[6])
        sk_r2s.append(results[7])
        dataset_names.append(dataset_name)
    display(Markdown(f"**{model_name} - Time Comparison**"))
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(dataset_names))
    ax.bar(x, my_times, width=0.4, label='My Model', align='center', color='skyblue')
    ax.bar([i + 0.4 for i in x], sk_times, width=0.4, label='SK Model', align='center', color='lightgreen')
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Time Comparison of My Model vs SK Model on Different Datasets")
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(dataset_names)
    ax.legend()
    plt.show()
    display(Markdown(f"**{model_name} - MAE Comparison**"))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, my_maes, width=0.4, label='My Model', align='center', color='skyblue')
    ax.bar([i + 0.4 for i in x], sk_maes, width=0.4, label='SK Model', align='center', color='lightgreen')
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("MAE")
    ax.set_title("MAE Comparison of My Model vs SK Model on Different Datasets")
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(dataset_names)
    ax.legend()
    plt.show()
    display(Markdown(f"**{model_name} - RMSE Comparison**"))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, my_rmses, width=0.4, label='My Model', align='center', color='skyblue')
    ax.bar([i + 0.4 for i in x], sk_rmses, width=0.4, label='SK Model', align='center', color='lightgreen')
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Comparison of My Model vs SK Model on Different Datasets")
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(dataset_names)
    ax.legend()
    plt.show()
    display(Markdown(f"**{model_name} - R² Comparison**"))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, my_r2s, width=0.4, label='My Model', align='center', color='skyblue')
    ax.bar([i + 0.4 for i in x], sk_r2s, width=0.4, label='SK Model', align='center', color='lightgreen')
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("R²")
    ax.set_title("R² Comparison of My Model vs SK Model on Different Datasets")
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(dataset_names)
    ax.legend()
    plt.show()

def regression_predictions(my_model):
    _, new, target = regression_data()
    predictions = my_model.predict(new)
    new_with_predictions = new.copy()
    new_with_predictions[target + " (Predicted)"] = predictions
    model_name = get_model_name(my_model)
    display(Markdown(f"**{model_name} - Predictions on New Data**"))
    return new_with_predictions

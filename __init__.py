import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split

# def evaluate_regression(df, target, my_model, sk_model):
#     X_train, X_test, Y_train, Y_test = split_data(df, target=target)
#     start_time = time.time()
#     my_model.fit(X_train, Y_train)
#     my_Y_pred = my_model.predict(X_test)
#     my_time = time.time() - start_time
#     start_time = time.time()
#     sk_model.fit(X_train, Y_train)
#     sk_Y_pred = sk_model.predict(X_test)
#     sk_time = time.time() - start_time
#     display_model_name(my_model)
#     print(f"{'':<5} {'My Model':<25} {'SK Model':<25}")
#     print(f"{'Time':<5} {my_time:<25} {sk_time:<25}")
#     print(f"{'MAE':<5} {mean_absolute_error(Y_test, my_Y_pred):<25} {mean_absolute_error(Y_test, sk_Y_pred):<25}")
#     print(f"{'RMSE':<5} {root_mean_squared_error(Y_test, my_Y_pred):<25} {root_mean_squared_error(Y_test, sk_Y_pred):<25}")
#     print(f"{'R2':<5} {r2_score(Y_test, my_Y_pred):<25} {r2_score(Y_test, sk_Y_pred):<25}")
    
    
# def evaluate_minmaxscaler(df, my_model, sk_model):
#     X_train = df
#     start_time = time.time()
#     my_model.fit_transform(X_train)
#     my_time = time.time() - start_time
#     start_time = time.time()
#     sk_model.fit_transform(X_train)
#     sk_time = time.time() - start_time
#     display_model_name(my_model)
#     print(f"{'':<5} {'My Model':<25} {'SK Model':<25}")
#     print(f"{'Time':<5} {my_time:<25} {sk_time:<25}")


def display_model_name(model):
    model_name = ' '.join(re.findall(r'[A-Z][a-z]*', type(model).__name__))
    display(Markdown(f"**{model_name}**"))
    
def split_data(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=[target]).values
    Y = df[target].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test

# =============================== CLASSIFICATION =============================== #
def classification_data():
    classification_datasets = {
        "Classification Big": pd.read_csv('classification_big.csv'),
        "Classification Medium": pd.read_csv('classification_medium.csv'),
        "Classification Small": pd.read_csv('classification_small.csv')
    }
    classification_new = pd.read_csv('classification_new.csv'),
    classification_target = 'revenue'
    return classification_datasets, classification_new, classification_target

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
    display_model_name(my_model)
    print(f"{'':<5} {'My Model':<25} {'SK Model':<25}")
    print(f"{'Time':<5} {my_time:<25} {sk_time:<25}")
    print(f"{'Acc':<5} {accuracy_score(Y_test, my_Y_pred):<25} {accuracy_score(Y_test, sk_Y_pred):<25}")
    print(f"{'F1':<5} {f1_score(Y_test, my_Y_pred, average='weighted'):<25} {f1_score(Y_test, sk_Y_pred, average='weighted'):<25}")
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

def classification_evaluates(datasets, target, my_model, sk_model):
    my_times = []
    sk_times = []
    dataset_names = []
    for dataset_name, df in datasets.items():
        print(f"\nEvaluating on {dataset_name} dataset:")
        my_time, sk_time = classification_evaluate(df, target, my_model, sk_model)
        my_times.append(my_time)
        sk_times.append(sk_time)
        dataset_names.append(dataset_name)
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
    

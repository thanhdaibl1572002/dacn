import time
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from IPython.display import display, Markdown

def split_data(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=[target]).values
    Y = df[target].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test

def evaluate_regression(df, target, my_model, sk_model):
    X_train, X_test, Y_train, Y_test = split_data(df, target=target)
    start_time = time.time()
    my_model.fit(X_train, Y_train)
    my_Y_pred = my_model.predict(X_test)
    my_time = time.time() - start_time
    start_time = time.time()
    sk_model.fit(X_train, Y_train)
    sk_Y_pred = sk_model.predict(X_test)
    sk_time = time.time() - start_time
    display(Markdown(f"**{' '.join(re.findall('[A-Z][a-z]*', type(my_model).__name__))}**"))
    print(f"{'':<5} {'My Model':<25} {'SK Model':<25}")
    print(f"{'Time':<5} {my_time:<25} {sk_time:<25}")
    print(f"{'MAE':<5} {mean_absolute_error(Y_test, my_Y_pred):<25} {mean_absolute_error(Y_test, sk_Y_pred):<25}")
    print(f"{'RMSE':<5} {root_mean_squared_error(Y_test, my_Y_pred):<25} {root_mean_squared_error(Y_test, sk_Y_pred):<25}")
    print(f"{'R2':<5} {r2_score(Y_test, my_Y_pred):<25} {r2_score(Y_test, sk_Y_pred):<25}")

def evaluate_classification(df, target, my_model, sk_model):
    X_train, X_test, Y_train, Y_test = split_data(df, target=target)
    start_time = time.time()
    my_model.fit(X_train, Y_train)
    my_Y_pred = my_model.predict(X_test)
    my_time = time.time() - start_time
    start_time = time.time()
    sk_model.fit(X_train, Y_train)
    sk_Y_pred = sk_model.predict(X_test)
    sk_time = time.time() - start_time
    display(Markdown(f"**{' '.join(re.findall('[A-Z][a-z]*', type(my_model).__name__))}**"))
    print(f"{'':<5} {'My Model':<25} {'SK Model':<25}")
    print(f"{'Time':<5} {my_time:<25} {sk_time:<25}")
    print(f"{'Acc':<5} {accuracy_score(Y_test, my_Y_pred):<25} {accuracy_score(Y_test, sk_Y_pred):<25}")
    print(f"{'F1':<5} {f1_score(Y_test, my_Y_pred, average='weighted'):<25} {f1_score(Y_test, sk_Y_pred, average='weighted'):<25}")
    cm_my = confusion_matrix(Y_test, my_Y_pred)
    cm_sk = confusion_matrix(Y_test, sk_Y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm_my, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y_test), yticklabels=np.unique(Y_test), ax=axes[0])
    axes[0].set_title('Confusion Matrix for My Model')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    sns.heatmap(cm_sk, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y_test), yticklabels=np.unique(Y_test), ax=axes[1])
    axes[1].set_title('Confusion Matrix for SK Model')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    plt.tight_layout()
    plt.show()
    accuracy_scores = [accuracy_score(Y_test, my_Y_pred), accuracy_score(Y_test, sk_Y_pred)]
    f1_scores = [f1_score(Y_test, my_Y_pred, average='weighted'), f1_score(Y_test, sk_Y_pred, average='weighted')]
    models = ['My Model', 'SK Model']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x=models, y=accuracy_scores, ax=axes[0], palette='Blues')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    sns.barplot(x=models, y=f1_scores, ax=axes[1], palette='Greens')
    axes[1].set_title('F1 Score Comparison')
    axes[1].set_ylabel('F1 Score')
    plt.tight_layout()
    plt.show()
    
def evaluate_minmaxscaler(df, my_model, sk_model):
    X_train = df
    start_time = time.time()
    my_model.fit_transform(X_train)
    my_time = time.time() - start_time
    start_time = time.time()
    sk_model.fit_transform(X_train)
    sk_time = time.time() - start_time
    display(Markdown(f"**{' '.join(re.findall('[A-Z][a-z]*', type(my_model).__name__))}**"))
    print(f"{'':<5} {'My Model':<25} {'SK Model':<25}")
    print(f"{'Time':<5} {my_time:<25} {sk_time:<25}")
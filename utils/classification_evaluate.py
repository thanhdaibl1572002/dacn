import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils.split_data import split_data
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
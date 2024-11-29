import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from dacn.utils.split_data import split_data

def classification_evaluate(df, target, my_model, sk_model, learning_curve=False):
    X_train, X_test, Y_train, Y_test = split_data(df, target=target)
    start_time = time.time()
    if learning_curve: my_model.fit(X_train, Y_train, X_test, Y_test)
    else: my_model.fit(X_train, Y_train)
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
    my_f1_score = f"{f1_score(Y_test, my_Y_pred, average='macro'):.3f}"
    sk_f1_score = f"{f1_score(Y_test, sk_Y_pred, average='macro'):.3f}"
    print(f"\n{'':<5} {'My Model':<25} {'SK Model':<25}")
    print(f"{'Time':<5} {my_time_s:<25} {sk_time_s}")
    print(f"{'Acc':<5} {my_acc_percent:<25} {sk_acc_percent}")
    print(f"{'F1':<5} {my_f1_score:<25} {sk_f1_score}\n")
    # Learning Curve
    if learning_curve:
        epochs_to_plot = my_model.epochs_to_plot
        train_losses = my_model.train_losses
        test_losses = my_model.test_losses
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_to_plot, train_losses, marker='o', label='Train Loss')
        plt.plot(epochs_to_plot, test_losses, marker='x', label='Test Loss')
        for i, loss in enumerate(train_losses):
            plt.text(epochs_to_plot[i], loss, f"{loss:.4f}", ha='center', va='bottom', fontsize=9)
        for i, loss in enumerate(test_losses):
            plt.text(epochs_to_plot[i], loss, f"{loss:.4f}", ha='center', va='bottom', fontsize=9)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Learning Curve (Train vs Test Loss)")
        plt.legend()
        plt.grid(True)
        plt.show()
        epochs_to_plot.clear()
        train_losses.clear()
        test_losses.clear()
    # Confusion Matrix
    cm_my = confusion_matrix(Y_test, my_Y_pred)
    cm_sk = confusion_matrix(Y_test, sk_Y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm_my, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title('Confusion Matrix for My Model')
    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')
    sns.heatmap(cm_sk, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
    axes[1].set_title('Confusion Matrix for SK Model')
    axes[1].set_xlabel('Predicted Labels')
    axes[1].set_ylabel('True Labels')
    plt.tight_layout()
    plt.show()
    # Classification Report
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
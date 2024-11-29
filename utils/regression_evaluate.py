import time
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score
from dacn.utils.split_data import split_data
# def regression_evaluate(df, target, my_model, sk_model):
#     X_train, X_test, Y_train, Y_test = split_data(df, target=target)
#     start_time = time.time()
#     my_model.fit(X_train, Y_train)
#     my_Y_pred = my_model.predict(X_test)
#     my_time = time.time() - start_time
#     start_time = time.time()
#     sk_model.fit(X_train, Y_train)
#     sk_Y_pred = sk_model.predict(X_test)
#     sk_time = time.time() - start_time
#     my_time_s = f"{my_time:.3f} s"
#     sk_time_s = f"{sk_time:.3f} s"
#     my_mae = f"{mean_absolute_error(Y_test, my_Y_pred):.3f}"
#     sk_mae = f"{mean_absolute_error(Y_test, sk_Y_pred):.3f}"
#     my_rmse = f"{root_mean_squared_error(Y_test, my_Y_pred):.3f}"
#     sk_rmse = f"{root_mean_squared_error(Y_test, sk_Y_pred):.3f}"
#     my_mse = f"{mean_squared_error(Y_test, my_Y_pred):.3f}"
#     sk_mse = f"{mean_squared_error(Y_test, sk_Y_pred):.3f}"
#     my_r2 = f"{r2_score(Y_test, my_Y_pred):.3f}"
#     sk_r2 = f"{r2_score(Y_test, sk_Y_pred):.3f}"
#     print(my_model.loss)
#     print(f"\n{'':<5} {'My Model':<25} {'SK Model':<25}")
#     print(f"{'Time':<5} {my_time_s:<25} {sk_time_s}")
#     print(f"{'MAE':<5} {my_mae:<25} {sk_mae}")
#     print(f"{'RMSE':<5} {my_rmse:<25} {sk_rmse}")
#     print(f"{'MSE':<5} {my_mse:<25} {sk_mse}")
#     print(f"{'R2':<5} {my_r2:<25} {sk_r2}\n")
#     return my_time, sk_time, my_mae, sk_mae, my_rmse, sk_rmse, my_mse, sk_mse, my_r2, sk_r2


def regression_evaluate(df, target, my_model, sk_model, learning_curve=False):
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
    my_time_s = f"{my_time:.3f} s"
    sk_time_s = f"{sk_time:.3f} s"
    my_mae = f"{mean_absolute_error(Y_test, my_Y_pred):.3f}"
    sk_mae = f"{mean_absolute_error(Y_test, sk_Y_pred):.3f}"
    my_rmse = f"{root_mean_squared_error(Y_test, my_Y_pred):.3f}"
    sk_rmse = f"{root_mean_squared_error(Y_test, sk_Y_pred):.3f}"
    my_mse = f"{mean_squared_error(Y_test, my_Y_pred):.3f}"
    sk_mse = f"{mean_squared_error(Y_test, sk_Y_pred):.3f}"
    my_r2 = f"{r2_score(Y_test, my_Y_pred):.3f}"
    sk_r2 = f"{r2_score(Y_test, sk_Y_pred):.3f}"
    print(f"\n{'':<5} {'My Model':<25} {'SK Model':<25}")
    print(f"{'Time':<5} {my_time_s:<25} {sk_time_s}")
    print(f"{'MAE':<5} {my_mae:<25} {sk_mae}")
    print(f"{'RMSE':<5} {my_rmse:<25} {sk_rmse}")
    print(f"{'MSE':<5} {my_mse:<25} {sk_mse}")
    print(f"{'R2':<5} {my_r2:<25} {sk_r2}\n")
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
    # MAE Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['My Model'], [float(my_mae)], width=0.4, label='My Model', color='skyblue')
    ax.bar(['SK Model'], [float(sk_mae)], width=0.4, label='SK Model', color='lightgreen')
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Models")
    ax.set_ylabel("MAE")
    ax.set_title("MAE Comparison of My Model vs SK Model")
    ax.legend()
    plt.show()
    # RMSE Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['My Model'], [float(my_rmse)], width=0.4, label='My Model', color='skyblue')
    ax.bar(['SK Model'], [float(sk_rmse)], width=0.4, label='SK Model', color='lightgreen')
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Models")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Comparison of My Model vs SK Model")
    ax.legend()
    plt.show()
    # MSE Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['My Model'], [float(my_mse)], width=0.4, label='My Model', color='skyblue')
    ax.bar(['SK Model'], [float(sk_mse)], width=0.4, label='SK Model', color='lightgreen')
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Models")
    ax.set_ylabel("MSE")
    ax.set_title("MSE Comparison of My Model vs SK Model")
    ax.legend()
    plt.show()
    # R² Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['My Model'], [float(my_r2)], width=0.4, label='My Model', color='skyblue')
    ax.bar(['SK Model'], [float(sk_r2)], width=0.4, label='SK Model', color='lightgreen')
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Models")
    ax.set_ylabel("R²")
    ax.set_title("R² Comparison of My Model vs SK Model")
    ax.legend()
    plt.show()
    return my_time, sk_time
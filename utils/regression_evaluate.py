import time
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from dacn.utils.split_data import split_data
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
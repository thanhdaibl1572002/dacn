import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score

def measure_time(model, X_train, Y_train, mode="fit"):
    start_time = time.time()
    if mode == "fit_transform":
        model.fit_transform(X_train)
    else:
        model.fit(X_train, Y_train)
    return time.time() - start_time

def evaluate(df, target, my_model, sk_model, task_type="regression"):
    if task_type == "minmaxscaler":
        X_train = df
        Y_train =  None
        my_time = measure_time(my_model, X_train, Y_train, 'fit_transform')
        sk_time = measure_time(sk_model, X_train, Y_train, 'fit_transform')
        print(f"{'':<5} {'My Model':<25} {'SK Model':<25}")
        print(f"{'Time':<5} {my_time:<25} {sk_time:<25}")
    else:
        X = df.drop(columns=[target]).values
        Y = df[target].values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        my_time = measure_time(my_model, X_train, Y_train)
        sk_time = measure_time(sk_model, X_train, Y_train)
        print(f"{'':<5} {'My Model':<25} {'SK Model':<25}")
        print(f"{'Time':<5} {my_time:<25} {sk_time:<25}")
        my_Y_pred = my_model.predict(X_test)
        sk_Y_pred = sk_model.predict(X_test)
        if task_type == "regression":
            print(f"{'MAE':<5} {mean_absolute_error(Y_test, my_Y_pred):<25} {mean_absolute_error(Y_test, sk_Y_pred):<25}")
            print(f"{'RMSE':<5} {root_mean_squared_error(Y_test, my_Y_pred):<25} {root_mean_squared_error(Y_test, sk_Y_pred):<25}")
            print(f"{'R2':<5} {r2_score(Y_test, my_Y_pred):<25} {r2_score(Y_test, sk_Y_pred):<25}")
        elif task_type == "classification":
            print(f"{'Acc':<5} {accuracy_score(Y_test, my_Y_pred):<25} {accuracy_score(Y_test, sk_Y_pred):<25}")
            print(f"{'F1':<5} {f1_score(Y_test, my_Y_pred, average='weighted'):<25} {f1_score(Y_test, sk_Y_pred, average='weighted'):<25}")
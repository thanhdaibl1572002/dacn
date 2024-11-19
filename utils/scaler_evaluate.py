import time
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
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
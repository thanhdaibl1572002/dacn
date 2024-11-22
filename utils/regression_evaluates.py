import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from dacn.utils.regression_data import regression_data
from dacn.utils.regression_evaluate import regression_evaluate
from dacn.utils.model_name import model_name
def regression_evaluates(my_model, sk_model):
    datasets, _, target = regression_data()
    my_times = []
    sk_times = []
    my_maes = []
    sk_maes = []
    my_rmses = []
    sk_rmses = []
    my_mses = []
    sk_mses = []
    my_r2s = []
    sk_r2s = []
    dataset_names = []
    my_model_name = model_name(my_model)
    for dataset_name, df in datasets.items():
        display(Markdown(f"**{my_model_name} - {dataset_name}**"))
        results = regression_evaluate(df, target, my_model, sk_model)
        my_times.append(results[0])
        sk_times.append(results[1])
        my_maes.append(results[2])
        sk_maes.append(results[3])
        my_rmses.append(results[4])
        sk_rmses.append(results[5])
        my_mses.append(results[6])
        sk_mses.append(results[7])
        my_r2s.append(results[8])
        sk_r2s.append(results[9])
        dataset_names.append(dataset_name)
    display(Markdown(f"**{my_model_name} - Time Comparison**"))
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
    display(Markdown(f"**{my_model_name} - MAE Comparison**"))
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
    display(Markdown(f"**{my_model_name} - RMSE Comparison**"))
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
    display(Markdown(f"**{my_model_name} - MSE Comparison**"))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, my_mses, width=0.4, label='My Model', align='center', color='skyblue')
    ax.bar([i + 0.4 for i in x], sk_mses, width=0.4, label='SK Model', align='center', color='lightgreen')
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("MSE")
    ax.set_title("MSE Comparison of My Model vs SK Model on Different Datasets")
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(dataset_names)
    ax.legend()
    plt.show()
    display(Markdown(f"**{my_model_name} - R² Comparison**"))
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
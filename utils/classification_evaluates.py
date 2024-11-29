import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from dacn.utils.classification_data import classification_data
from dacn.utils.classification_evaluate import classification_evaluate
from dacn.utils.model_name import model_name
def classification_evaluates(my_model, sk_model, learning_curve=False):
    datasets, _, target = classification_data()
    my_times = []
    sk_times = []
    dataset_names = []
    my_model_name = model_name(my_model)
    for dataset_name, df in datasets.items():
        display(Markdown(f"**{my_model_name} - {dataset_name}**"))
        my_time, sk_time = classification_evaluate(df, target, my_model, sk_model, learning_curve)
        my_times.append(my_time)
        sk_times.append(sk_time)
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
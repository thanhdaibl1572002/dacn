from IPython.display import display, Markdown
from dacn.utils.regression_data import regression_data
from dacn.utils.model_name import model_name
def regression_predictions(my_model):
    _, new, target = regression_data()
    predictions = my_model.predict(new)
    new_with_predictions = new.copy()
    new_with_predictions[target + " (Predicted)"] = predictions
    my_model_name = model_name(my_model)
    display(Markdown(f"**{my_model_name} - Predictions on New Data**"))
    return new_with_predictions
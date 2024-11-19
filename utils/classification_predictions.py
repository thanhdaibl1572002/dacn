from IPython.display import display, Markdown
from utils.classification_data import classification_data
from utils.model_name import model_name
def classification_predictions(my_model):
    _, new, target = classification_data()
    predictions = my_model.predict(new)
    new_with_predictions = new.copy()
    new_with_predictions[target + " (Predicted)"] = predictions
    my_model_name = model_name(my_model)
    display(Markdown(f"**{my_model_name} - Predictions On New Data**"))
    return new_with_predictions
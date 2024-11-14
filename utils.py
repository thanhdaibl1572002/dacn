import re
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split

def display_model_name(model):
    model_name = ' '.join(re.findall(r'[A-Z][a-z]*', type(model).__name__))
    display(Markdown(f"**{model_name}**"))
    
def split_data(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=[target]).values
    Y = df[target].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test
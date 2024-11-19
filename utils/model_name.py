import re
def model_name(model):
    return ' '.join(re.findall(r'[A-Z][a-z]*', type(model).__name__))
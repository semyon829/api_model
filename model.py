import pickle

PATH_TO_MODELS = 'models/'
filename = 'model.pkl'

def load_model(filename):
    model = PATH_TO_MODELS + filename
    loaded_model = pickle.load(open(model, 'rb'))
    return loaded_model
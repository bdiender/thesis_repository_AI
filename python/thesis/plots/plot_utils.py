import pickle


def load_model(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
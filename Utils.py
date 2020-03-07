from sklearn.externals import joblib


def load_model(load_model_name: str):

    model_path = 'Models/' + load_model_name + '.m'
    model = joblib.load(model_path)

    return model

from joblib import dump

def save_model(model, path, filename, domain_params, model_params, losses):
    """
    Saves the model as pkl

    @param model: NN model, for this work shallow network sampled using SWIM algorithm of type sklearn Pipeline
    @param path: path to save model
    @param filename: name of the file to save the model
    @param domain_params: domain parameters that is used to train the model
    @param model_params: model parameters that is used to train the model
    @param losses: losses of the model
    """
    dump(model, path + filename)
    dump(domain_params, path + 'DOMAIN_PARAMS_' + filename)
    dump(model_params, path + 'MODEL_PARAMS_' + filename)
    dump(losses, path + 'LOSSES_' + filename)

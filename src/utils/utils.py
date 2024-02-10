from activations import relu, tanh, sigmoid, elu, identity, gaussian, gelu, silu, softplus
from hamiltonians import single_pendulum, lotka_volterra, double_pendulum
from error_functions import mae, mse, l2_error, l2_error_relative
import numpy as np

def parse_activation(activation):
    """
    Given activation function string, returns the activation function and its derivative

    @param activation str from list [ 'relu', 'leaky_relu', 'elu', 'tanh', 'sigmoid', 'gaussian', 'gelu', 'silu', 'softplus' ]

    @returns activation function and its derivative
    """
    match activation:
        case "relu":
            return relu.relu, relu.d_relu
        case "leaky_relu":
            return relu.leaky_relu, relu.d_leaky_relu
        case "elu":
            return elu.elu, elu.d_elu
        case "tanh":
            return tanh.tanh, tanh.d_tanh
        case "sigmoid":
            return sigmoid.sigmoid, sigmoid.d_sigmoid
        case "gaussian":
            return gaussian.gaussian, gaussian.d_gaussian
        case "gelu":
            return gelu.gelu, gelu.d_gelu
        case "silu":
            return silu.silu, silu.d_silu
        case "softplus":
            return softplus.softplus, softplus.d_softplus
        case _:
            # default to identity
            return identity.identity, identity.d_identity

def parse_system_name(system_name):
    """
    @param system_name: Hamiltonian name as string

    @returns Hamiltonian function and its derivative
    """
    match system_name:
        case "single_pendulum":
            system = single_pendulum.SinglePendulum(m=1, l=1, g=1)
        case "lotka_volterra":
            system = lotka_volterra.LotkaVolterra()
        case "double_pendulum":
            system = double_pendulum.DoublePendulum()
        case _:
            raise ValueError("System not defined")

    return system.H, system.dH

def parse_error_function(error_function_name):
    """
    @param eror_function_name   : Error function name as string

    @returns Error function
    """
    match error_function_name:
        case "mean_absolute_error":
            return mae.mean_absolute_error
        case "mean_squared_error":
            return mse.mean_squared_error
        case "l2_error":
            return l2_error.l2_error
        case "l2_error_relative":
            return l2_error_relative.l2_error_relative
        case _:
            raise ValueError("Error function not defined")

def get_errors(y_true, y_pred):
    """
    @param y_true   :   true values
    @param y_pred   :   predictions

    @returns {mae, mse, l2_error, l2_error_relative}
    """
    return {
        'mean_absolute_error': mae.mean_absolute_error(y_true, y_pred),
        'mean_squared_error': mse.mean_squared_error(y_true, y_pred),
        'l2_error': l2_error.l2_error(y_true, y_pred),
        'l2_error_relative': l2_error_relative.l2_error_relative(y_true, y_pred),
    }

def get_results(experiment, model="ELM", dataset="train", type="errors", error_function="mean_absolute_error"):
    """
    @param experiment : Experiment object consisting of multiple runs
    @param model      : "ELM", "U-SWIM", "A-SWIM"
    @param dataset    : "train", "test"
    @param type       : "errors", "losses" where "errors" are calculated over the true and predicted H(x) values, "losses" are calculated over the true and predicted derivatives of H(x)
    @param function   : "mean_absolute_error", "mean_squared_error", "l2_error", "l2_error_relative"

    @returns list of results
    """
    return [ run[dataset + '_' + type][model][error_function] for run in experiment['runs']]

def get_summary(experiment, models, datasets, types, error_functions, stats):
    summary = []
    for dataset in datasets:
        # e.g. train
        for type in types:
            # e.g. losses
            for error_function in error_functions:
                # e.g. l2_error_relative
                for stat in stats:
                    summary.append(' '.join([dataset, type, stat, f'({error_function})']))
                    for model in models:
                        # e.g. 'ELM'
                        results = get_results(experiment, model, dataset, type, error_function)
                        match stat:
                            case 'min':
                                error = np.min(results)
                            case 'median':
                                error = np.median(results)
                            case 'mean':
                                error = np.mean(results)
                            case 'max':
                                error = np.max(results)
                            case _:
                                raise ValueError("Unknown stat")

                        summary.append(f'- {model}   \t: {str(error)}')
    return "\n".join(summary)

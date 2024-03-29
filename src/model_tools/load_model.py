import os
import json
import torch
from src.functions.aux_functions import pickPoolLayer

from src.models.LeNetPlus import LeNetPlus
from src.models.SupervisedNiNPlus import SupervisedNiNPlus
from src.models.DenseNetPlus import DenseNetPlus
from src.models.RegNet import RegNetX_200MF, RegNetX_400MF, RegNetY_400MF


PATH_MODELS = os.path.join('..', '..', 'reports', 'models')
default_params_file = os.path.join('..', '..', 'config', 'default_parameters.json')


def return_if_available(dictionary, original_val, key):
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return original_val


def load_model(file_name, model_type, info_file_name=None, info_data=None):

    # WARNING: Changes in the network definition may (and most certainly will) break the compatibility between previous
    # saved models and this function, since the state_dict structure will possibly vary:

    # WARNING: load_model assumes that the model to be loaded was trained using a cuda device.

    model = None

    if info_data is None:
        if info_file_name is not None:
            info_file_path = os.path.join(PATH_MODELS, info_file_name)
            info_file = open(info_file_path, mode='r')
            info_data = json.load(info_file)
            info_file.close()
        else:
            raise Exception('Neither info_data dict nor info_file_name provided.')

    pool_type = info_data['pool_type']
    if 'global_pool_type' not in info_data.keys():
        # Added for backwards compatibility
        global_pool_type = 'gap'
    else:
        global_pool_type = info_data['global_pool_type']
    pool_layer = pickPoolLayer(pool_type)
    global_pool_layer = pickPoolLayer(global_pool_type)
    if model_type == 'lenet':
        input_size = info_data['input_size']
        num_classes = info_data['num_classes']
        use_batch_norm = info_data['use_batch_norm']
        pool_aggrs = info_data['pool_aggrs']
        model = LeNetPlus(input_size[0], num_classes, pool_layer=pool_layer, use_batch_norm=use_batch_norm, aggregations=pool_aggrs)
    elif model_type == 'nin':
        input_size = info_data['input_size']
        num_classes = info_data['num_classes']
        pool_aggrs = info_data['pool_aggrs']
        model = SupervisedNiNPlus(pool_layer, global_pool_layer=global_pool_layer, in_channels=input_size[-1], num_classes=num_classes, input_size=input_size[:-1], aggregations=pool_aggrs)
    elif model_type == 'dense100':
        input_size = info_data['input_size']
        num_classes = info_data['num_classes']
        pool_aggrs = info_data['pool_aggrs']
        model = DenseNetPlus(pool_layer=pool_layer, global_pool_layer=global_pool_layer, in_channels=input_size[-1], num_classes=num_classes, num_layers=100, aggregations=pool_aggrs)
    elif model_type == 'regnet_x_200mf':
        num_classes = info_data['num_classes']
        pool_aggrs = info_data['pool_aggrs']
        model = RegNetX_200MF(global_pool_layer=global_pool_layer, num_classes=num_classes, aggregations=pool_aggrs)
    elif model_type == 'regnet_x_400mf':
        num_classes = info_data['num_classes']
        pool_aggrs = info_data['pool_aggrs']
        model = RegNetX_400MF(global_pool_layer=global_pool_layer, num_classes=num_classes, aggregations=pool_aggrs)
    elif model_type == 'regnet_y_400mf':
        num_classes = info_data['num_classes']
        pool_aggrs = info_data['pool_aggrs']
        model = RegNetY_400MF(global_pool_layer=global_pool_layer, num_classes=num_classes, aggregations=pool_aggrs)
    else:
        raise Exception('{} model type unavailable.'.format(model_type))
    # Load the state_dict of the model into the newly created model (load the learnt parameters):
    file_path = os.path.join(PATH_MODELS, file_name)
    if torch.cuda.is_available():
        state_dict = torch.load(file_path)
    else:
        # NOTE: We assume that all tensor data is stored in "cuda" format by default:
        # Convert the data saved in cuda format to cpu:
        state_dict = torch.load(file_path, map_location={'cuda:0': 'cpu'})
    model.load_state_dict(state_dict)
    return model

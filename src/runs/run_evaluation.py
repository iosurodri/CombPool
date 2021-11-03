import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

# Data loading and saving:
from src.data.load_data import load_dataset

# Model interaction:
from src.model_tools.evaluate import get_prediction_metrics
from src.model_tools.load_model import load_model

# Auxiliar modules
import torch
import json
import argparse

PATH_MODEL = os.path.join('..', '..', 'reports', 'models')


def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("test_name", nargs=1, type=str, help='Name of the model to be tested')
    CLI.add_argument("--test_idx", nargs="?", type=int, help='If provided, specifies the index of the test model to be loaded.')
    CLI.add_argument("--batch_size", nargs="?", type=int, default=32, help="""If provided, sets the number of samples to be loaded
        at once as part of a data batch.""")
    return CLI.parse_args()


def run_evaluation(test_name=None, test_idx=None, batch_size=32):

    # Models are assumed to be stored according to the following directory structure:
    #   /reports
    #       /models
    #           /test_name
    #               * test_0
    #               * test_0_info.json
    #               * ...
    #               * test_N
    #               * test_N_info.json
    # By default, test_0 will be loaded (unless specified otherwise through --test_idx)

    
    if test_idx is None:
        test_idx = 0
    file_name = os.path.join(test_name, 'test_{}'.format(str(test_idx)))
    model_file = os.path.join(PATH_MODEL, file_name)
    info_model_file = model_file + '_info.json'

    info_data = {}
    with open(info_model_file) as config_file:
        config_data = json.load(config_file)
        # Train loop configuration:
        input_size = config_data['input_size']
        info_data['input_size'] = input_size
        num_classes = config_data['num_classes']
        info_data['num_classes'] = num_classes
        model_type = config_data['model_type']
        info_data['model_type'] = model_type
        dataset = config_data['dataset']
        use_batch_norm = config_data['use_batch_norm']
        info_data['use_batch_norm'] = use_batch_norm
        pool_type = config_data['pool_type']
        info_data['pool_type'] = pool_type
        pool_aggrs = config_data['pool_aggrs']
        info_data['pool_aggrs'] = pool_aggrs

    # Check if a "cuda" device is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the required dataset
    test_loader = load_dataset(dataset, batch_size=batch_size, train=False, num_workers=0, pin_memory=True)
    # Load the required model
    model = load_model(file_name, model_type, info_data=info_data).to(device)
    # Evaluate the given model
    prediction_metrics = get_prediction_metrics(model, device=device, test_loader=test_loader)
    print(prediction_metrics)


if __name__ == '__main__':

    if len(sys.argv) == 2:
        # The following instruction unrolls all values separated by spaces of the second argument. Useful when this
        # arg is read as a string (as our bash gnu parallel script does), to allow the proper work of argparse:
        sys.argv = [sys.argv[0], *sys.argv[1].split()]
    args = parse_args()
    test_name = args.test_name[0]
    test_idx = args.test_idx
    batch_size = args.batch_size

    run_evaluation(test_name, test_idx, batch_size)

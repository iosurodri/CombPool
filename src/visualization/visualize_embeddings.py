import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

import torch
import argparse
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# Data loading and saving:
from src.data.load_data import load_dataset

# Model interaction:
from src.model_tools.load_model import PATH_MODELS, load_model

PATH_RUNS = os.path.join('..', '..', 'reports', 'runs')
PATH_MODELS = os.path.join('..', '..', 'reports', 'models')

def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--name", nargs="?", type=str, help='Name for the generated files. If none, a name based on the '
                                                         'current date and time will be used instead')
    CLI.add_argument("--dataset", nargs="?", type=str, default="CIFAR10", help='Dataset to be used for training. Options'
                                                                               'are "CIFAR10" for CIFAR10 dataset; '
                                                                               'Defaults to "CIFAR10".')
    CLI.add_argument("--test_idx", nargs="?", type=int, help='Index for the test to be run')    
    CLI.add_argument("--batch_size", nargs="?", type=int, default=32, help="""If provided, sets the number of samples to be loaded
        at once as part of a data batch.""")
    return CLI.parse_args()


def get_model(test_name, test_idx=None):

    if test_idx is None:
        test_idx = 0
    file_name = os.path.join(test_name, 'test_{}'.format(str(test_idx)))
    model_file = os.path.join(PATH_MODELS, file_name)
    info_model_file = model_file + '_info.json'

    # info_data = {}
    # with open(info_model_file) as config_file:
    #     config_data = json.load(config_file)
    #     # Train loop configuration:
    #     input_size = config_data['input_size']
    #     info_data['input_size'] = input_size
    #     num_classes = config_data['num_classes']
    #     info_data['num_classes'] = num_classes
    #     model_type = config_data['model_type']
    #     info_data['model_type'] = model_type
    #     dataset = config_data['dataset']
    #     use_batch_norm = config_data['use_batch_norm']
    #     info_data['use_batch_norm'] = use_batch_norm
    #     pool_type = config_data['pool_type']
    #     info_data['pool_type'] = pool_type
    #     pool_aggrs = config_data['pool_aggrs']
    #     info_data['pool_aggrs'] = pool_aggrs

    # Check if a "cuda" device is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the required model
    # model = load_model(file_name, model_type, info_data=info_data).to(device)
    model = load_model(file_name, model_type='dense100', info_file_name=info_model_file).to(device)
    return model


def visualize_embeddings(model, dataloader, name_test, add_imgs=False, max_samples=1000, device='cuda'):

    embedding_folder = os.path.join(PATH_RUNS, name_test + '_embedding')
    try:
        os.mkdir(embedding_folder)
    except FileExistsError:
        pass
    writer = SummaryWriter(log_dir=embedding_folder)

    if add_imgs:
        all_images = None
    all_labels = []
    all_features = None

    model.eval()

    sample_count = 0
    with torch.no_grad():
        # for data in dataloader:

        for i, data in enumerate(tqdm(dataloader, unit='batches', leave=False), 0):

            samples, labels = data[0], data[1].tolist()
            if add_imgs:
                if all_images is None:
                    all_images = samples
                else:
                    all_images = torch.cat([all_images, samples], dim=0)
            samples = samples.to(device)
            # Iterate over all the layers of the model until reaching the classifier:
            model_children = model.named_children()
            name_module, module = next(model_children)
            features = samples
            while 'classifier' not in name_module:
                features = module(features)
                name_module, module = next(model_children)
            # ToDo: Behaviour specific to DenseNet model
            features = F.relu(features, inplace=True)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            # features = module(features)
            if all_features is None:
                all_features = features.cpu()
            else:
                all_features = torch.cat([all_features, features.cpu()], dim=0)
            all_labels.extend(labels)
            # Check if enough images have been registered
            sample_count += len(samples)
            if sample_count > max_samples:
                break
        if add_imgs:
            writer.add_embedding(all_features, metadata=all_labels, tag='extracted_features', label_img=all_images)
        else:
            writer.add_embedding(all_features, metadata=all_labels, tag='extracted_features')
    writer.close()


if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        # The following instruction unrolls all values separated by spaces of the second argument. Useful when this
        # arg is read as a string (as our bash gnu parallel script does), to allow the proper work of argparse:
        sys.argv = [sys.argv[0], *sys.argv[1].split()]
    
    args = parse_args()
    test_name = args.name
    dataset = args.dataset
    test_idx = args.test_idx
    batch_size = args.batch_size
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the required dataset
    _, val_loader = load_dataset(dataset, batch_size=32, train=True, val=True, num_workers=0, pin_memory=True)
    
    model = get_model(test_name, test_idx=0)
    
    visualize_embeddings(model, val_loader, test_name, device=device, max_samples=1000, add_imgs=True)

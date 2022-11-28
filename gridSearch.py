import json
import os
from random import uniform
import argparse
from train_ae import train_example
from utils.dataset import ShapeNetDataset
from visualization_tools import printPointCloud as ptPC
import torch
import sys
from sklearn.model_selection import ParameterGrid
import numpy as np
import gc

def optimize_params(filepath=os.path.join("parameters", "params.json")):
    """
    :param filepath: string: json file path (contains ALL the hyperparameters, also those fixed: see
        lr_params.json for reference).
        N.B.: it should not contain the hyperparameters passed through default_params
    :return:
    """
    json_params = json.loads(open(filepath).read())
    parser = argparse.ArgumentParser(description=f'Model validation')
    args = parser.parse_args()
    best_val_loss = sys.float_info.max
    dict_params = {}
    best_hyperparams = {}  # contains the best hyperparameters (only those randomly generated) {hyperparam: value, ...}
    current_hyperparams = {}  # contains the best hyperparameters (only those randomly generated)
    param_grid = {}
    for hyperparam, value in json_params.items():
        # check if 'value' is a list
        if isinstance(value, list):
            param_grid[hyperparam] = value
        else:
            if value == 'None':
                value = None
            setattr(args, hyperparam, value)

    test_dataset = ShapeNetDataset(
        root=args.dataset,
        split='test',
        class_choice=args.test_class_choice,
        npoints=1024)

    counter = 0
    for current_param_grid in ParameterGrid(param_grid):
        for hyperparam, value in current_param_grid.items():
            setattr(args, hyperparam, value)
            current_hyperparams[hyperparam] = value
        setattr(args, 'runNumber', counter)
        print(f"\n\n------------------------------------------------------------------\nParameters: {args}\n")
        # val_losses is the list of losses obtained during validation
        model, val_losses = train_example(args)
        if min(val_losses) < best_val_loss:
            print(f"--- Best validation loss found! {min(val_losses)} (previous one: {best_val_loss}), corresponding to "
                  f"hyperparameters {current_hyperparams.items()}")
            best_val_loss = min(val_losses)
            best_hyperparams = current_hyperparams
        ptPC.print_original_decoded_point_clouds(test_dataset, args.test_class_choice, model, args)
        #dict_params[hash(str(args))] = str(args)
        dict_params[counter] = str(args)
        counter = counter + 1
    folder = args.outf
    try:
        os.makedirs(folder)
    except OSError:
        pass
    with open(os.path.join(folder, f'params_dictionary.json'), 'w') as f:
        json.dump(dict_params, f)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best_hyperparams


# def wrapper_train_by_class(json_path=os.path.join("parameters", "params.json")):
#     json_params = json.loads(open(json_path).read())
#     parser = argparse.ArgumentParser(description=f'Training models with different classes')
#     args = parser.parse_args()
#     for hyperparam, value in json_params.items():
#         if value == 'None':
#             value = None
#         setattr(args, hyperparam, value)
#     train_model_by_class(args)


if __name__ == '__main__':
    best_params = optimize_params(filepath=os.path.join("parameters", "gridsearch_table_parameters.json"))
    #print(f"Best parameters: \t{best_params}\n")
    # wrapper_train_by_class()


import argparse
import json
import os
import random

import numpy as np
import torch
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
import matplotlib
import math


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def upload_args_from_json(file_path=os.path.join("parameters", "fixed_params.json")):
    parser = argparse.ArgumentParser(description=f'Arguments from json')
    args = parser.parse_args()
    json_params = json.loads(open(file_path).read())
    for option, option_value in json_params.items():
        if option_value == 'None':
            option_value = None
        setattr(args, option, option_value)
    setattr(args, "runNumber", 0)
    return args


def farthest_and_nearest_points(x, batch_points, n):
    # x should be (batch_size, num_points, 3)
    # batch_points should be (batch_size, 1, 3)
    batch_size = x.size(0)
    num_points = x.size(1)
    assert batch_points.size(0) == batch_size
    assert batch_points.size(2) == 3
    assert batch_points.size(1) == 1
    # batch_points: for each batch (point cloud) we have one point (x,y,z)
    distances = torch.sum((x - batch_points) ** 2, dim=-1)
    idx = distances.topk(k=num_points, dim=-1)[1]
    idx_farthest = idx[:, :n]
    idx_nearest = idx[:, n:]
    idx_base = torch.arange(0, batch_size, device="cuda").view(-1, 1) * num_points
    idx_farthest = (idx_farthest + idx_base).view(-1)
    idx_nearest = (idx_nearest + idx_base).view(-1)
    x = x.view(-1, 3)
    x_far = x[idx_farthest, :]
    x_near = x[idx_nearest, :]
    return x_far.view(batch_size, n, 3), idx_farthest, x_near.view(batch_size, -1, 3)


def plot_neptune_losses(folder, chosen_loss="cd", log_scale=True):
    # filename format:
    # for training: <cd or accuracy>___training__numSpheres-<n>_optionFc-<True or False>
    # for validation: <cd or accuracy>___validation__numSpheres-<n>_optionFc-<True or False>
    matplotlib.rcParams["lines.linewidth"] = .5
    colors = ["blue", "g", "y", "r", "black", "fuchsia"]
    loss_names = {
        "cd": "Mean Chamfer Loss",
        "accuracy": "Accuracy"
    }
    count_color = 0
    min_loss_description = ""
    dict_files = dict()
    sns.set_style("darkgrid")
    min_loss = 100000000
    max_loss = 0
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        type_loss, file_description = filename.split("___")
        run_info = file_description.split("__")[1]
        file_description = file_description.replace("__", ": ").replace("_", ", ").replace(".csv", "").replace("-", ": ")
        if not os.path.isfile(filepath) or type_loss != chosen_loss:
            continue
        if dict_files.get(run_info, 0) == 0:
            color = colors[count_color]
            count_color += 1
            dict_files[run_info] = color
        else:
            color = dict_files[run_info]
        losses = pd.read_csv(filepath, header=None)
        losses = losses[[0, 2]]
        losses.columns = ["epoch", "loss"]
        if file_description.startswith("val"):
            linestyle = "--"
            min_loss_description = file_description if min(losses["loss"]) < min_loss else min_loss_description
        else:
            linestyle = None
        sns.lineplot(x="epoch", y="loss", data=losses, linestyle=linestyle, color=color, label=file_description)
        min_loss = min(min(losses["loss"]), min_loss)
        max_loss = max(max(losses["loss"]), max_loss)
    plt.gca().set_xlabel("Epoch", fontsize=12)
    plt.gca().set_ylabel(loss_names[type_loss], fontsize=12)
    plt.gca().set_title("OnionNet: Training and Validation Losses", fontsize=14)
    if log_scale:
        sample_count = np.around(np.logspace(math.log10(min_loss), math.log10(max_loss), 8), 4)
        plt.gca().set(yscale='log')
        plt.gca().set(yticks=sample_count, yticklabels=sample_count)
        plt.gca().set_ylabel(f"{loss_names[type_loss]} (log scale)", fontsize=12)
    print(f"Min Loss: {min_loss}, params: {min_loss_description}")
    plt.savefig(f"train_val_{type_loss}_loss.png", bbox_inches="tight", linewidth=.2)


def cropping(batch_point_cloud, batch_target=None, num_cropped_points=512, fixed_choice=None):
    # batch_point_cloud: (batch_size, num_points, 3)
    batch_size = batch_point_cloud.size(0)
    num_points = batch_point_cloud.size(1)
    k = num_points - num_cropped_points
    choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]), torch.Tensor([-1, 0, 0]),
              torch.Tensor([-1, 1, 0])] if fixed_choice is None else [fixed_choice]
    p_center = []
    for m in range(batch_size):
        index = random.sample(choice, 1)
        p_center.append(index[0])
    # it should have shape (batch_size, 3): one center for one point_cloud inside the batch
    batch_centers = torch.cat(p_center, dim=0).cuda()
    batch_centers = batch_centers.view(-1, 1, 3)
    incomplete_input, idx, cropped_input = farthest_and_nearest_points(batch_point_cloud, batch_centers, k)
    if batch_target is not None:
        batch_target = batch_target.view(-1, 1)
        batch_target = batch_target[idx, :]
        return incomplete_input, batch_target.view(-1, k, 1), cropped_input
    return incomplete_input, cropped_input


if __name__ == "__main__":
    plot_neptune_losses("../grid_search_files")

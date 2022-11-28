import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec
import utils.dataset as ds
import torch

from point_completion.multitask_ext_code_model import OnionNet
from utils.dataset_seg import ShapeNetPart

# Insert the cloud path in order to print it
from utils.utils import upload_args_from_json, cropping


def printCloudFile(cloud_original, cloud_decoded, name):
    alpha = 0.5
    xyz = np.loadtxt(cloud_original)
    # print(xyz)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', alpha=alpha)
    ax.set_title("original cloud")
    ax.append(fig.add_subplot(111, projection='3d'))

    ax[-1].plot(cloud_decoded[:, 0], cloud_decoded[:, 1], cloud_decoded[:, 2], 'o', alpha=alpha)
    ax[-1].set_title("decoded cloud")
    plt.savefig(f"/content/pointnet.pytorch/{name}.png")


def printCloud(cloud_original, name, alpha=0.5, opt=None):
    xyz = cloud_original[0]
    print("sono qui")
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', alpha=alpha)
    ax.set_title("original cloud")
    folder = "/content/pointnet.pytorch/images/" if opt is None else os.path.join(opt.outf, "images")
    try:
        os.makedirs(folder)
    except OSError:
        pass
    plt.savefig(os.path.join(folder, f"{hash(str(opt))}_{name}.png"))


def printCloudM(cloud_original, cloud_decoded, name, alpha=0.5, opt=None):
    xyz = cloud_original[0]
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', alpha=alpha)
    ax.set_title("original cloud")
    xyz = cloud_decoded[0]
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', alpha=alpha)
    ax.set_title("decoded cloud")
    folder = "/content/pointnet.pytorch/images/" if opt is None else os.path.join(opt.outf, "images")
    try:
        os.makedirs(folder)
    except OSError:
        pass
    plt.savefig(os.path.join(folder, f"{hash(str(opt))}_{name}.png"))


def savePtsFile(type, category, opt, array, run=None):
    folder = os.path.join(opt.outf, "visualizations", f"{opt.runNumber}", category)
    string_neptune_path = f"point_clouds/{category}/{type}"
    try:
        os.makedirs(folder)
    except OSError:
        pass
    pc_file = open(os.path.join(folder, f"{type}.pts"), 'w')
    np.savetxt(pc_file, array)
    if run is not None:
        run[string_neptune_path].upload(pc_file.name)
    pc_file.close()


def print_original_decoded_point_clouds(dataset, category, model, opt, run=None):
    categories = [category]
    if category is None:
        categories = dataset.get_categories()
    for category in categories:
        n_point_clouds = 30 if hasattr(opt, 'novel_categories') else 10
        for index in range(n_point_clouds):
            point_cloud = dataset.get_point_cloud_by_category(category, index=index)
            model.eval()
            point_cloud_np = point_cloud.cuda()
            point_cloud_np = torch.unsqueeze(point_cloud_np, 0)
            decoded_point_cloud = model(point_cloud_np)
            if opt.type_decoder == "pyramid":
                decoded_point_cloud = decoded_point_cloud[2]  # take only the actual input reconstruction
            original_pc_np = point_cloud_np.cpu().numpy()
            decoded_pc_np = decoded_point_cloud.cpu().data.numpy()
            # printCloudM(point_cloud_np, dec_val_stamp, name=category, opt=opt)
            original_pc_np = original_pc_np.reshape((1024, 3))
            decoded_pc_np = decoded_pc_np.reshape((1024, 3))
            savePtsFile(f"original_n{index}", category, opt, original_pc_np, run)
            savePtsFile(f"decoded_n{index}", category, opt, decoded_pc_np, run)


def print_original_incomplete_decoded_point_clouds(category, model, opt, run):
    categories = ["airplane", "car", "chair", "lamp", "mug", "motorbike", "table"]\
        if category is None else [category]
    model.eval()
    centers = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]), torch.Tensor([-1, 0, 0]),
              torch.Tensor([-1, 1, 0])]
    for category in categories:
        dataset = ShapeNetPart(root=opt.dataset, class_choice=category, split="test")
        for index in range(6):
            if index > dataset.__len__():
                break
            point_cloud = dataset[index]
            point_cloud = point_cloud.cuda()
            point_cloud = torch.unsqueeze(point_cloud, 0)
            original_pc_np = point_cloud.cpu().numpy()
            original_pc_np = original_pc_np.reshape((-1, 3))
            savePtsFile(f"{index}_original", category, opt, original_pc_np, run)
            for num_crop, center in enumerate(centers):
                incomplete_cloud, _ = cropping(point_cloud, fixed_choice=center)
                incomplete_cloud = incomplete_cloud.cuda()
                if opt.segmentation:
                    decoded_point_cloud, pred = model(incomplete_cloud)
                    decoded_point_cloud = decoded_point_cloud[2]
                else:
                    decoded_point_cloud = model(incomplete_cloud)
                decoded_pc_np = decoded_point_cloud.cpu().data.numpy()
                incomplete_pc_np = incomplete_cloud.cpu().data.numpy()
                incomplete_pc_np = incomplete_pc_np.reshape((-1, 3))
                decoded_pc_np = decoded_pc_np.reshape((-1, 3))
                savePtsFile(f"{index}_{num_crop}_incomplete", category, opt, incomplete_pc_np, run)
                savePtsFile(f"{index}_{num_crop}_decoded", category, opt, decoded_pc_np, run)


def init_radius(num_spheres, constant_area=False):
    r_max = np.sqrt(3)
    if constant_area:
        coeffs = np.array([np.power(2, i / 3) for i in range(num_spheres)])
        r1 = r_max / coeffs[-1]
        radius_array = r1 * coeffs
    else:
        radius_array = r_max * np.arange(1, num_spheres + 1) / num_spheres
    return torch.tensor(radius_array).view(radius_array.shape[0], 1, 1), r_max


def spherical_features(x, pred, num_spheres, num_classes, radius_tensor, r_max):
    batch_size = x.size(0)
    num_radius = radius_tensor.size(0)
    pred = pred.view(batch_size, x.size(1), 1)
    distances_from_origin = torch.sqrt(torch.sum(x ** 2, dim=-1))
    # point_belongs_to: each point is associated to a specific sphere (from 0 up to num_spheres-1)
    # distances_from_origin =  - radius_tensor
    dfo_m_r = radius_tensor - distances_from_origin.repeat(num_radius, 1, 1)
    dfo_m_r[dfo_m_r < 0] = r_max
    point_belongs_to = torch.min(dfo_m_r, dim=0)[1]
    id_and_pred = torch.cat((point_belongs_to.view(batch_size, -1, 1), pred.view(batch_size, -1, 1)), dim=-1)
    feat = []
    for id_sphere in range(num_spheres):
        for id_batch in range(batch_size):
            current_sphere_feat = torch.bincount(
                id_and_pred[id_batch, point_belongs_to[id_batch, :] == id_sphere, :][:, -1].to(torch.int),
                minlength=num_classes)
            tot_frequency = torch.sum(current_sphere_feat)
            current_sphere_feat = current_sphere_feat / tot_frequency.item() if tot_frequency.item() != 0 else current_sphere_feat
            feat.append(current_sphere_feat.view(1, num_classes))
    return torch.cat(feat, dim=-1).view(batch_size, -1), id_and_pred



def print_onion_net(opt, num_spheres=5):
    dataset_airplane = ShapeNetPart(root=opt.dataset, segmentation=True, class_choice="airplane", split="test")
    centers = [torch.Tensor([1, 0, 0])]
    i = 0
    radius_tensor, r_max = init_radius(num_spheres)
    radius_tensor = radius_tensor
    num_classes = 27
    airplane, seg_airplane = dataset_airplane[i]
    airplane, seg_airplane = airplane.view(1, -1, 3), seg_airplane.view(1, -1),
    incomplete_airplane, seg_airplane, _ = cropping(airplane, seg_airplane, fixed_choice=centers[i], device="cpu")
    _, id_and_pred_airplane = spherical_features(incomplete_airplane, seg_airplane, num_spheres, num_classes, radius_tensor, r_max)
    for sphere in range(num_spheres):
        mask_curr_sphere_airplane = id_and_pred_airplane[:, :, -2] == sphere
        if mask_curr_sphere_airplane.sum() == 0:
            continue
        filt_pc_airplane_in_sphere = incomplete_airplane[mask_curr_sphere_airplane]
        for seg_class in range(dataset_airplane.seg_num_class[0]):
            mask_curr_seg_class = id_and_pred_airplane[mask_curr_sphere_airplane][:, -1] == seg_class
            if mask_curr_seg_class.sum() == 0:
                continue
            filt_pc_airplane = filt_pc_airplane_in_sphere[mask_curr_seg_class]
            savePtsFile(f"sphere{sphere}_seg{seg_class}", "airplane", opt, filt_pc_airplane.cpu().numpy(), None)


if __name__=="__main__":
    file = "D:\\UNIVERSITA\\PRIMO ANNO\\SECONDO SEMESTRE\\Machine learning and Deep learning\\PROJECTS\\P1\\CODICE\\parameters\\pc_fixed_params.json"
    opt = upload_args_from_json(file)
    setattr(opt, "runNumber", 0)
    print_onion_net(opt)
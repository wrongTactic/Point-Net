from __future__ import print_function
from pointnet.pointnet_model import PointNet_AutoEncoder
from pointnet.deeper_pointnet_model import PointNet_DeeperAutoEncoder
from gcnn.gcnn_model import DGCNN_AutoEncoder
from utils.loss import PointLoss
import argparse
import torch.optim as optim
import torch.utils.data
from utils.dataset import ShapeNetDataset
import gc
from utils.early_stopping import EarlyStopping
from utils.FPS import farthest_point_sample, index_points
import json
from utils.utils import upload_args_from_json
from visualization_tools import printPointCloud
from visualization_tools.printPointCloud import *
import neptune.new as neptune


def evaluate_loss_by_class(opt, autoencoder, run):
    run["params"] = vars(opt)
    classes = ["Airplane", "Car", "Chair", "Lamp", "Mug", "Motorbike", "Table"] if opt.test_class_choice is None\
        else [opt.test_class_choice]
    autoencoder.cuda()
    print("Start evaluation loss by class")
    for classs in classes:
        print(f"\t{classs}")
        test_dataset = ShapeNetDataset(opt.dataset,
                                       opt.num_points,
                                       class_choice=classs,
                                       split='test')
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))
        run[f"loss/{classs}"] = test_example(opt, test_dataloader, autoencoder)
        print()
    if opt.test_class_choice is None:
        evaluate_novel_categories(opt, autoencoder, run)


def evaluate_novel_categories(opt, autoencoder, run):
    setattr(opt, "novel_categories", True)
    classes = ["Basket", "Bicycle", "Bookshelf", "Bottle", "Bowl", "Clock", "Helmet", "Microphone", "Microwave",
               "Pianoforte", "Rifle", "Telephone", "Watercraft"]
    print("Start evaluation novel categories...")
    for classs in classes:
        print(f"\t{classs}")
        novel_dataset = ShapeNetDataset("/content/drive/MyDrive/novel_categories", opt.num_points, class_choice=classs, split='test')
        novel_dataloader = torch.utils.data.DataLoader(
            novel_dataset,
            num_workers=int(opt.workers)
        )
        run[f"loss/novel_categories/{classs}"] = test_example(opt, novel_dataloader, autoencoder)
        print_original_decoded_point_clouds(novel_dataset, classs, autoencoder, opt, run)
        print()


def test_example(opt, test_dataloader, model):
    # initialize lists to monitor test loss and accuracy
    chamfer_loss = PointLoss()
    test_loss = 0.0

    model.eval()  # prep model for evaluation

    for data in test_dataloader:
        # forward pass: compute predicted outputs by passing inputs to the model
        data = data.cuda()
        output = model(data)
        if opt.type_decoder == "pyramid":
            output = output[2] #take only the actual prediction (not the sampling predictions)
        output = output.cuda()
        # calculate the loss
        loss = chamfer_loss(data, output)
        # update test loss
        test_loss += loss.item() * data.size(0)

    # calculate and print avg test loss
    test_loss = test_loss / len(test_dataloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    return test_loss


def train_example(opt):
    neptune_info = json.loads(open(os.path.join("parameters", "neptune_params.json")).read())
    run = neptune.init(project=neptune_info['project'],
                       tags=[str(opt.type_encoder), str(opt.train_class_choice), str(opt.size_encoder)],
                   api_token=neptune_info['api_token'])
    run['params'] = vars(opt)
    random_seed = 43
    torch.manual_seed(random_seed)
    # writer = SummaryWriter('runs/train_ae_experiment_1')
    training_dataset = ShapeNetDataset(
        root=opt.dataset,
        class_choice=opt.train_class_choice,
        npoints=opt.num_points,
        set_size=opt.set_size)

    validation_dataset = ShapeNetDataset(
        root=opt.dataset,
        split='val',
        class_choice=opt.train_class_choice,
        npoints=opt.num_points,
        set_size=opt.set_size)

    final_training = opt.final_training
    if final_training:
        if opt.runNumber == 0:
            print("!!!!!!Final training starts!!!!!!")
        test_dataset = ShapeNetDataset(
            root=opt.dataset,
            split='test',
            class_choice=opt.test_class_choice,
            npoints=opt.num_points,
            set_size=opt.set_size)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))
        training_dataset = torch.utils.data.ConcatDataset([training_dataset, validation_dataset])

    train_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    val_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))


    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    if opt.type_encoder == "pointnet":
        autoencoder = PointNet_DeeperAutoEncoder(opt.num_points, opt.size_encoder, dropout=opt.dropout) \
            if opt.architecture == "deep" else \
            PointNet_AutoEncoder(opt, opt.num_points, opt.size_encoder, dropout=opt.dropout)
    elif opt.type_encoder == 'dgcnn':
        autoencoder = DGCNN_AutoEncoder(opt)
    else:
        raise IOError(f"Invalid type_encoder!! Should be 'pointnet' or 'dgcnn'. Found: {opt.type_encoder}")
    if opt.runNumber == 0 and opt.architecture == "deep":
        print("!!!!!!Training a deeper model!!!!!!")
    if opt.model != '':
        autoencoder.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(autoencoder.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2),
                           weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.scheduler_stepSize, gamma=opt.scheduler_gamma)
    autoencoder.cuda()
    run["model"] = autoencoder
    checkpoint_path = os.path.join(opt.outf, f"checkpoint{opt.runNumber}.pt")
    training_history = []
    val_history = []
    gc.collect()
    torch.cuda.empty_cache()
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, path=checkpoint_path)
    #  instantiate the loss
    chamfer_loss = PointLoss()
    n_epoch = opt.nepoch
    for epoch in range(n_epoch):
        if epoch > 0:
            scheduler.step()
        if epoch < 30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch < 80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2
        training_losses = []
        for i, points in enumerate(train_dataloader, 0):
            points = points.cuda()
            optimizer.zero_grad()
            autoencoder.train()
            decoded_points = autoencoder(points)
            # let's compute the chamfer distance between the two sets: 'points' and 'decoded'
            if opt.type_decoder == "pyramid":
                decoded_coarse = decoded_points[0].cuda()
                decoded_fine = decoded_points[1].cuda()
                decoded_input = decoded_points[2].cuda()

                coarse_sampling_idx = farthest_point_sample(points, 128, RAN=False)
                coarse_sampling = index_points(points, coarse_sampling_idx)
                coarse_sampling = coarse_sampling.cuda()
                fine_sampling_idx = farthest_point_sample(points, 256, RAN=True)
                fine_sampling = index_points(points, fine_sampling_idx)
                fine_sampling = fine_sampling.cuda()

                CD_loss = chamfer_loss(points, decoded_input)

                loss = chamfer_loss(points, decoded_input) \
                       + alpha1 * chamfer_loss(coarse_sampling, decoded_coarse) \
                       + alpha2 * chamfer_loss(fine_sampling, decoded_fine)
            else:
                decoded_points = decoded_points.cuda()
                CD_loss = loss = chamfer_loss(points, decoded_points)
            training_losses.append(CD_loss.item())
            run["train/batch_loss"].log(CD_loss.item())
            loss.backward()
            optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()
        train_mean = np.average(training_losses)
        run["train/epoch_loss"].log(train_mean)

        # Validation Phase
        if not final_training:
            with torch.no_grad():
                val_losses = []
                for j, val_points in enumerate(val_dataloader, 0):
                    autoencoder.eval()
                    val_points = val_points.cuda()
                    decoded_val_points = autoencoder(val_points)
                    if opt.type_decoder == "pyramid":
                        decoded_val_points = decoded_val_points[2]  # take only the actual prediction (num_points)
                    decoded_val_points = decoded_val_points.cuda()
                    val_loss = chamfer_loss(val_points, decoded_val_points)
                    val_losses.append(val_loss.item())
                    run["validation/batch_loss"].log(val_loss.item())
                val_mean = np.average(val_losses)
                run["validation/epoch_loss"].log(val_mean)

                print(f'\tepoch: {epoch}, training loss: {train_mean}, validation loss: {val_mean}')
        else:
            print(f'\tepoch: {epoch}, training loss: {train_mean}')

        if epoch >= 50:
            early_stopping(val_mean if not final_training else train_mean, autoencoder)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        training_history.append(train_mean)
        if not final_training:
            val_history.append(val_mean)

    if opt.nepoch <= 50:
        torch.save(autoencoder.state_dict(), checkpoint_path)
    autoencoder.load_state_dict(torch.load(checkpoint_path))
    printPointCloud.print_original_decoded_point_clouds(ShapeNetDataset(
        root=opt.dataset,
        split='test',
        class_choice=opt.test_class_choice,
        npoints=opt.num_points,
        set_size=opt.set_size), opt.test_class_choice, autoencoder, opt, run)
    if not final_training:
        run.stop()
        return autoencoder, val_history
    else:
        run["model_dictionary"].upload(checkpoint_path)
        evaluate_loss_by_class(opt, autoencoder, run)
        run.stop()
        return autoencoder, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    opt = upload_args_from_json(os.path.join("parameters", "ae_fixed_params.json"))
    print(f"\n\n------------------------------------------------------------------\nParameters: {opt}\n")
    train_example(opt)

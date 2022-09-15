# -*- coding: utf-8 -*-
import math

import config
import helper
import uuid
import matplotlib.pyplot as plt

import os
import torch
import numpy as np

import GNN

import time
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

np.random.seed(1000)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
MODEL_WEIGHT_BACKUP_PATH = "./output"
MODEL_RES_PATH = "./res"
DEEP_CBT_SAVE_PATH = "./output/deep_cbts"
TEMP_FOLDER = "./temp"


def show_image(img, i, score, res_path, model_name):
    img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
    plt.imshow(img)
    plt.title("fold " + str(i) + " Frobenious distance: " + "{:.2f}".format(score))
    plt.axis('off')
    plt.savefig(res_path + model_name + "_cbt_" + str(i) + ".png")
    plt.show()


def generate_subject_biased_cbts(model, train_data):
    """
        Generates all possible CBTs for a given training set.
        Args:
            model: trained Dual-HINet model
            train_data: list of data objects
    """
    model.eval()
    cbts = np.zeros((config.N_Nodes, config.N_Nodes, len(train_data)))
    Ss = []
    train_data = [d.to(device) for d in train_data]
    for i, data in enumerate(train_data):
        cbt, S = model(data)
        cbts[:, :, i] = np.array(cbt.cpu().detach())
        Ss.append([s.cpu().detach().numpy() for s in S])
    Ss = np.array(Ss, dtype=object)

    return cbts, Ss


def generate_cbt_median(model, train_data):
    """
        Generate optimized CBT for the training set (use post training refinement)
        Args:
            model: trained Dual-HINet model
            train_data: list of data objects
    """
    model.eval()
    cbts = []
    train_data = [d.to(device) for d in train_data]
    for data in train_data:
        cbt, _ = model(data)
        cbts.append(np.array(cbt.cpu().detach()))
    final_cbt = torch.tensor(np.median(cbts, axis=0), dtype=torch.float32).to(device)

    return final_cbt


def mean_frobenious_distance(generated_cbt, test_data):
    """
        Calculate the mean Frobenious distance between the CBT and test subjects (all views)
        Args:
            generated_cbt: trained Dual-HINet model
            test_data: list of data objects
    """
    frobenius_all = []
    for data in test_data:
        views = data.con_mat
        for index in range(views.shape[2]):
            diff = torch.abs(views[:, :, index] - generated_cbt)
            diff = diff * diff
            sum_of_all = diff.sum()
            d = torch.sqrt(sum_of_all)
            frobenius_all.append(d)
    return sum(frobenius_all) / len(frobenius_all)


def mean_distance_between_multigraphs(multigraphs):
    frobenius_all = []  # frob_sum = 0
    N = len(multigraphs)
    k = 0
    for i in range(N):
        for j in range(N - k):
            if i != j:
                # frob_view_sum = 0
                for index in range(config.N_views):
                    diff = torch.abs(multigraphs[i][:, :, index] - multigraphs[j][:, :, index])
                    diff = diff * diff
                    sum_of_all = diff.sum()
                    d = torch.sqrt(sum_of_all)
                    frobenius_all.append(d)  # frob_view_sum += d#
                # frobenius_all.append(frob_view_sum)
        k += 1
    return sum(frobenius_all) / len(frobenius_all)


def S_loss(Ss, samples):
    total_s_dist = 0
    for S in Ss:
        clustered_samples = []
        # S = S.detach().numpy()
        for views in samples:
            pooled_views = [S.T @ views[:, :, index] @ S for index in range(views.shape[2])]
            clustered_samples.append(torch.stack(pooled_views, -1))
        total_s_dist += mean_distance_between_multigraphs(clustered_samples)
        samples = clustered_samples
    return total_s_dist


def mae_to_subjects(generated_cbt, test_data):
    """
        Calculate the mean Frobenious distance between the CBT and test subjects (all views)
        Args:
            generated_cbt: trained Dual-HINet model
            test_data: list of data objects
    """
    MAEs = []
    for data in test_data:
        frobenius = []
        views = data.con_mat
        for index in range(views.shape[2]):
            diff = torch.abs(views[:, :, index] - generated_cbt)
            diff = diff * diff
            sum_of_all = diff.sum()
            d = torch.sqrt(sum_of_all)
            frobenius.append(d)
        MAEs.append(sum(frobenius) / len(frobenius))
    return MAEs


def train_model(X, model_params, n_max_epochs, early_stop, model_name, random_sample_size=10, n_folds=5):
    """
        Trains a model for each cross validation fold and
        saves all models along with CBTs to ./output/<model_name>
        Args:
            X (np array): dataset (train+test) with shape [N_Subjects, N_ROIs, N_ROIs, N_Views]
            n_max_epochs (int): number of training epochs (if early_stop == True this is maximum epoch limit)
            early_stop (bool): if set true, model will stop training when overfitting starts.
            model_name (string): name for saving the model
            random_sample_size (int): random subset size for SNL function
            n_folds (int): number of cross validation folds
        Return:
            models: trained models
    """
    list_of_losses_tracked = []  # List of tracked losses.
    list_of_rep_loss = []  # List of rep_loss

    models = []

    save_path = MODEL_WEIGHT_BACKUP_PATH + "/" + config.model_name + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save fold rep losses and their mean and std
    res_path = MODEL_RES_PATH + "/" + config.model_name + "/"
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    model_id = str(uuid.uuid4())
    with open(save_path + "model_params.txt", 'w') as f:
        print(model_params, file=f)
    with open(res_path + "model_params.txt", 'w') as f:
        print(model_params, file=f)

    N_views = config.N_views
    N_ROIs = config.N_Nodes

    CBTs = []
    MAEs = []
    scores = []
    for i in range(n_folds):
        torch.cuda.empty_cache()
        print("********* FOLD {} *********".format(i))
        train_data, test_data, train_mean, train_std = helper.preprocess_data_array(X, number_of_folds=n_folds,
                                                                                    current_fold_id=i)

        test_casted = [d.to(device) for d in helper.cast_data(test_data)]
        loss_weights = torch.tensor(np.array(list((1 / train_mean) / np.max(1 / train_mean)) * len(train_data)),
                                    dtype=torch.float32)
        loss_weights = loss_weights.to(device)
        train_casted = [d.to(device) for d in helper.cast_data(train_data)]

        train_targets = [torch.tensor(tensor, dtype=torch.float32).to(device) for tensor in train_data]

        test_errors = []
        tick = time.time()

        if model_params["num_pooling"] != 0:
            assign_ratio = math.pow(model_params["final_num_clusters"] / N_ROIs, 1 / model_params["num_pooling"])
        else:
            assign_ratio = 1
        model = GNN.SoftPoolingGcnEncoder(max_num_nodes=N_ROIs,
                                          input_dim=model_params["input_dim"],
                                          hidden_dim=model_params["hidden_dim"],
                                          embedding_dim=model_params["embedding_dim"],
                                          num_layers=model_params["num_layers"],
                                          assign_hidden_dim=model_params["hidden_dim"], view_dim=N_views,
                                          assign_ratio=assign_ratio, num_pooling=model_params["num_pooling"],
                                          bn=True, not_ablated=model_params["not_ablated"])
        model = model.to(device)

        params = list(model.parameters())

        optimizer = torch.optim.Adam(params, lr=model_params["learning_rate"], weight_decay=0.00)

        for epoch in range(n_max_epochs):
            model.train()
            losses = []

            for data in train_casted:
                cbt, S = model(data)

                # Centrality loss
                views_sampled = random.sample(train_targets, random_sample_size)
                sampled_targets = torch.cat(views_sampled, axis=2).permute((2, 1, 0))

                expanded_cbt = cbt.expand((sampled_targets.shape[0], model_params["N_ROIs"], model_params["N_ROIs"]))
                diff = torch.abs(expanded_cbt - sampled_targets)  # Absolute difference
                sum_of_all = torch.mul(diff, diff).sum(axis=(1, 2))  # Sum of squares
                l_c = torch.sqrt(sum_of_all)  # Square root of the sum
                l_c_norm = (l_c * loss_weights[:random_sample_size * model_params["n_attr"]]).sum()

                l_sum = l_c_norm

                # Clustering loss
                if model_params["is_joint_S_loss"]:
                    l_s = S_loss(S, views_sampled)

                    l_sum += model_params["S_loss_weight"] * l_s

                losses.append(l_sum)

            # Backprob
            optimizer.zero_grad()
            loss = torch.mean(torch.stack(losses))
            loss.backward()

            optimizer.step()

            # Track the loss
            if epoch % 10 == 0:
                cbt = generate_cbt_median(model, train_casted)
                rep_loss = mean_frobenious_distance(cbt, test_casted)

                tock = time.time()
                time_elapsed = tock - tick
                tick = tock
                rep_loss = float(rep_loss)
                test_errors.append(rep_loss)
                print(
                    "Epoch: {}  |  Test Rep: {:.2f}  |  Time Elapsed: {:.2f}  |".format(epoch, rep_loss, time_elapsed))
                # Early stopping control
                if len(test_errors) > 6 and early_stop:
                    torch.save(model.state_dict(),
                               TEMP_FOLDER + "/weight_" + model_id + "_" + str(rep_loss)[:5] + ".model")
                    last_6 = test_errors[-6:]
                    if (all(last_6[i] < last_6[i + 1] for i in range(5))):
                        print("Early Stopping")
                        break

        # Restore best model so far
        try:
            restore = "./temp/weight_" + model_id + "_" + str(min(test_errors))[:5] + ".model"
            model.load_state_dict(torch.load(restore))
        except:
            pass
        torch.save(model.state_dict(), save_path + "fold" + str(i) + ".model")
        models.append(model)

        # Generate and save refined CBT
        cbt = generate_cbt_median(model, train_casted)
        rep_loss = mean_frobenious_distance(cbt, test_casted)

        cbt = cbt.cpu().numpy()
        CBTs.append(cbt)
        np.save(save_path + "fold" + str(i) + "_cbt", cbt)
        np.save(res_path + "fold" + str(i) + "_cbt", cbt)
        # Save all subject biased CBTs
        all_cbts, all_S_matrices = generate_subject_biased_cbts(model, train_casted)
        np.save(save_path + "fold" + str(i) + "_all_cbts", all_cbts)
        np.save(res_path + "fold" + str(i) + "_all_cbts", all_cbts)
        np.save(res_path + "fold" + str(i) + "_all_S_matrices", all_S_matrices)
        scores.append(float(rep_loss))
        print("FINAL RESULTS  REP: {}".format(rep_loss))

        list_of_rep_loss.append(rep_loss)  # Fatih Said Duran 01.13.22

        mae_fold = mae_to_subjects(cbt, test_casted)
        MAEs = MAEs + mae_fold

        # Clean interim model weights
        helper.clear_dir(TEMP_FOLDER)

    print("List of rep losses:")
    for l_r_l in list_of_rep_loss:
        print(float(l_r_l), end=', ')
    print(np.mean(list_of_rep_loss))
    print(np.std(list_of_rep_loss))

    np.save(res_path + model_name + "_MAEs", MAEs)
    np.save(res_path + model_name + "_folds", list_of_rep_loss)


    for i, cbt in enumerate(CBTs):
        show_image(cbt, i, scores[i], res_path, model_name)
    return models

# For Dual-HINet, you can configure the model and training parameters here.

# Train on simulated data (S, normal random dist.) or external data (E)
Dataset = "S"

# Path for the dataset (binary file in MATLAB .mat format) with shape
# [N_Subjects, N_Nodes, N_Nodes, N_Views]
# ignored if Dataset = S
Path =  "./simulated dataset/example.mat"

# Number of simulated subjects (overwriten if Dataset = "E")
N_Subjects = 150
# Number of nodes for simulated brain networks (overwriten if Dataset = "E")
N_Nodes = 35
# Number of brain views (overwriten if Dataset = "E")
N_views = 4

# Number of training epochs
N_max_epochs = 100
# Apply early stopping True/False
early_stop =  True

# Random subset size for SNL function
random_sample_size = 10

# Number of cross validation folds
n_folds = 3

# Learning Rate for Adam optimizer
lr = 0.0005

# Dimensions of GCN modules
input_dim = 1
hidden_dim = 25
embedding_dim = 5

# Number of graph convolution layers in GCN module
num_layers = 3

# Number of layers of the clustering block
num_pooling = 1
# Final number of cluster nodes at the ned of clustering block
final_num_clusters = 3

# Should node-level block output node-embeddings be ablated
not_ablated = True

# Should clustering loss be included
is_S_loss = False
# Weight given to S loss while adding to total loss
S_loss_weight = 0.1

#Name of the model
model_name = "Dual-HINet"


#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
#                 Below is not to be modified manually                       #
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#

import numpy as np
import helper
from os import listdir
from os.path import isfile, join
import scipy.io

if Dataset.lower() not in ["e", "E", "s", "S"]:
    raise ValueError("Dataset options are E or S.")

if (Dataset.lower() == "e"):
    # Fatih Said Duran 01.07.22
    # ->
    matFiles = [f for f in listdir(Path) if isfile(join(Path, f))]
    ls_subjects = []
    for matfile in matFiles:
        views = scipy.io.loadmat(Path+matfile)['views']
        ls_subjects.append(views)
    X = np.stack(ls_subjects, axis=0)
    # <-
    #Above code is the replacement of the below line, to be able to load the .mat format real data
    #X = np.load(Path)

    N_Subjects = X.shape[0]
    N_Nodes = X.shape[1]
    N_views = X.shape[3]
else:
    X = helper.create_better_simulated(N_Subjects, N_Nodes) if N_views == 4 else helper.simulate_dataset(N_Subjects, N_Nodes, N_views)

CONFIG = {
        "X": X,
        "N_ROIs":  X.shape[1],
        "N_views":  X.shape[3],
        "N_max_epochs": N_max_epochs,
        "n_folds": n_folds,
        "random_sample_size": random_sample_size,
        "early_stop": early_stop,
        "model_name": model_name
    }

MODEL_PARAMS = {
        "N_ROIs": N_Nodes,
        "learning_rate" : lr,
        "n_attr": X.shape[3],

        "input_dim" : input_dim,
        "hidden_dim" : hidden_dim,
        "embedding_dim" : embedding_dim,

        "num_layers" : num_layers,

        "num_pooling" : num_pooling,
        "final_num_clusters" : final_num_clusters,

        "not_ablated" : not_ablated,

        "is_joint_S_loss" : is_S_loss,

        "S_loss_weight" : S_loss_weight,
    }

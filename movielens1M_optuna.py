### 2CDS 

print('2CDS Trials')

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dominating_set_algorithms import *

import optuna
from optuna.trial import TrialState

import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim, Tensor

from torch_sparse import SparseTensor, matmul

from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj


movie_path = './ml-1m/movies.dat'
rating_path = './ml-1m/ratings.dat'

from data_handling import *

user_mapping = load_node_csv(rating_path, index_col=0, header=None, delimiter='::', col_names=['movieId', 'rating',	'timestamp'], index_name='userId')
movie_mapping = load_node_csv(movie_path, index_col=0, header=None, delimiter='::', col_names=['title', 'genres'], index_name='movieId')

for key in movie_mapping:
    movie_mapping[key] += len(user_mapping)

print('Users: ', len(user_mapping), 'Items: ', len(movie_mapping))

edge_index = load_edge_csv(
    rating_path,
    src_index_col='userId',
    src_mapping=user_mapping,
    dst_index_col='movieId',
    dst_mapping=movie_mapping,
    link_index_col='rating',
    rating_threshold=3,
)

print('Edge index:', edge_index.shape)


# users = ['u' + str(u) for u in edge_index.tolist()[0]]
# items = ['i' + str(i) for i in edge_index.tolist()[1]]

# G = nx.Graph()
# G.add_nodes_from(users, bipartite=0)
# G.add_nodes_from(items, bipartite=1)
# G.add_edges_from(zip(users,items))

# print('Number of nodes: ', G.number_of_nodes())
# print('Number of edges: ', G.number_of_edges())


# dom_set = dominating_set(G,1,minimal=True,optimize=True)

# print('Number of nodes (ds): ', len(dom_set))
# print('Ratio (ds): ', len(dom_set) / G.number_of_nodes())

# ds_users, ds_movies = [], []

# for element in dom_set:
#   if element.startswith('u'):
#     ds_users.append(int(element[1:]))
#   elif element.startswith('i'):
#     ds_movies.append(int(element[1:]))

# print('(ds_users, ds_movies)', (len(ds_users), len(ds_movies)))

# ds_indices = torch.logical_or(torch.isin(edge_index[0], torch.tensor(ds_users)),
#                               torch.isin(edge_index[1], torch.tensor(ds_movies))).nonzero().reshape(-1).tolist()

# print('ds_indices:', len(ds_indices))


def import_txt(file_name):
    with open(file_name, 'r') as file:
        data = file.read().splitlines()
        data = [int(x) for x in data]

        return data

ds_indices = import_txt('c2_indices.txt')


num_users, num_movies = len(user_mapping), len(movie_mapping)
num_interactions = edge_index.shape[1]
all_indices = [i for i in range(num_interactions)]

train_indices, test_indices = train_test_split(
    all_indices, test_size=0.2, random_state=1)
val_indices, test_indices = train_test_split(
    test_indices, test_size=0.5, random_state=1)

train_indices = list(set(train_indices) & set(ds_indices))

train_edge_index = edge_index[:, train_indices]
val_edge_index = edge_index[:, val_indices]
test_edge_index = edge_index[:, test_indices]

('train_edge_index:', train_edge_index.shape)

train_sparse_edge_index = SparseTensor(row=torch.cat([train_edge_index[0],train_edge_index[1]]), 
                                       col=torch.cat([train_edge_index[1],train_edge_index[0]]), 
                                       sparse_sizes=(num_users + num_movies, num_users + num_movies))
val_sparse_edge_index = SparseTensor(row=torch.cat([val_edge_index[0],val_edge_index[1]]), 
                                     col=torch.cat([val_edge_index[1],val_edge_index[0]]), 
                                     sparse_sizes=(num_users + num_movies, num_users + num_movies))
test_sparse_edge_index = SparseTensor(row=torch.cat([test_edge_index[0],test_edge_index[1]]), 
                                      col=torch.cat([test_edge_index[1],test_edge_index[0]]), 
                                     sparse_sizes=(num_users + num_movies, num_users + num_movies))


from gcn_model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}.")

edge_index = edge_index.to(device)
train_edge_index = train_edge_index.to(device)
train_sparse_edge_index = train_sparse_edge_index.to(device)

val_edge_index = val_edge_index.to(device)
val_sparse_edge_index = val_sparse_edge_index.to(device)

train_edge_index_no_offset = train_edge_index.clone().detach()
train_edge_index_no_offset[1] = train_edge_index_no_offset[1] - len(user_mapping)

val_edge_index_no_offset = val_edge_index.clone().detach()
val_edge_index_no_offset[1] = val_edge_index_no_offset[1] - len(user_mapping)

test_edge_index = test_edge_index.to(device)
test_sparse_edge_index = test_sparse_edge_index.to(device)

test_edge_index_no_offset = test_edge_index.clone().detach()
test_edge_index_no_offset[1] = test_edge_index_no_offset[1] - len(user_mapping)

def objective(trial):
    # Generate the model.
    
    n_layers = trial.suggest_int("K", 1, 4)
    emb_dim = trial.suggest_categorical("embedding_dim", [32, 64, 96, 128])

    model = LightGCN(num_users, num_movies, embedding_dim=emb_dim, K=n_layers, add_self_loops=False).to(device)
    model.train()

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    LR = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # define contants
    ITERATIONS = 10000
    BATCH_SIZE = trial.suggest_int("batch_size", 512,2048)
    ITERS_PER_LR_DECAY = 200
    K = 20
    LAMBDA = trial.suggest_float("lambda_val", 1e-6, 1e-1, log=True)

    for iter in range(ITERATIONS):
        # forward propagation
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        train_sparse_edge_index)

        # mini batching
        user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
        BATCH_SIZE, train_edge_index_no_offset)
        user_indices, pos_item_indices, neg_item_indices = user_indices.to(
            device), pos_item_indices.to(device), neg_item_indices.to(device)
        users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
        pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
        neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]


        # loss computation
        train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                              pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
            scheduler.step()

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    model.eval()
    test_loss, test_recall, test_precision, test_ndcg = evaluation(
        model, test_edge_index_no_offset, test_sparse_edge_index, [train_edge_index_no_offset, val_edge_index_no_offset], K, LAMBDA)
        
    trial.report(test_recall, iter)

    return test_recall

study = optuna.create_study(direction="maximize",
                            storage="sqlite:///db_movielens_1M.sqlite3",  # Specify the storage URL here.
                            study_name="2CDS")
study.optimize(objective, n_trials=100, timeout=None)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
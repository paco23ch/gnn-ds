from data_handling import *
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split
import networkx as nx
import logging

# This creates a child logger that inherits from the root you configured above
logger = logging.getLogger(__name__)

def load_data(movie_path, rating_path):
    if movie_path.endswith('.dat'): 
        user_mapping = load_node_csv(rating_path, index_col=0, header=None, delimiter='::', col_names=['movieId', 'rating',	'timestamp'], index_name='userId')
        movie_mapping = load_node_csv(movie_path, index_col=0, header=None, delimiter='::', col_names=['title', 'genres'], index_name='movieId')
    else:
        user_mapping = load_node_csv(rating_path, index_col='userId')
        movie_mapping = load_node_csv(movie_path, index_col='movieId')

    for key in movie_mapping:
        movie_mapping[key] += len(user_mapping)

    logger.info(f'Users: {len(user_mapping)}, Items: {len(movie_mapping)}')

    edge_index = load_edge_csv(
        rating_path,
        src_index_col='userId',
        src_mapping=user_mapping,
        dst_index_col='movieId',
        dst_mapping=movie_mapping,
        link_index_col='rating',
        rating_threshold=3,
    )

    logger.info(f'Edge index: {edge_index.shape}')

    return(user_mapping, movie_mapping, edge_index)

def generate_graph(edge_index):
    users = ['u' + str(u) for u in edge_index.tolist()[0]]
    items = ['i' + str(i) for i in edge_index.tolist()[1]]

    G = nx.Graph()
    G.add_nodes_from(users, bipartite=0)
    G.add_nodes_from(items, bipartite=1)
    G.add_edges_from(zip(users,items))

    logger.info(f'Number of unique users: {len(set(users))}')
    logger.info(f'Number of unique items: {len(set(items))}')
    logger.info(f'Number of nodes: {G.number_of_nodes()}')
    logger.info(f'Number of edges: {G.number_of_edges()}')

    return G

def get_ds_edges(dom_set, edge_index, G):
    ds_users = [int(x[1:]) for x in dom_set if x[0]=='u']
    ds_movies = [int(x[1:]) for x in dom_set if x[0]=='i']

    ds_indices = torch.logical_or(torch.isin(edge_index[0], torch.tensor(ds_users)),
                                torch.isin(edge_index[1], torch.tensor(ds_movies))).nonzero().reshape(-1).tolist()
    
    logger.info(f'     Number of nodes: {len(dom_set)}')
    logger.info(f'     Ratio: {len(dom_set) / G.number_of_nodes()}')
    logger.info(f'     Number of unique users: {len(ds_users)}')
    logger.info(f'     Number of unique items: {len(ds_movies)}')
    logger.info(f'     Total dominant set edge: {len(ds_indices)}')
    logger.info(f'     Trainable edges: {len(ds_indices)*0.8}')

    return(ds_indices)

def get_users_items(dom_set):
    ds_users = [int(x[1:]) for x in dom_set if x[0]=='u']
    ds_items = [int(x[1:]) for x in dom_set if x[0]=='i']
    
    return [len(ds_users), len(ds_items)]

def split_data(edge_index):
    num_interactions = edge_index.shape[1]
    all_indices = [i for i in range(num_interactions)]

    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=1)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=1)

    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]

    return train_indices, val_indices, test_indices, train_edge_index, val_edge_index, test_edge_index

def get_sparse_tensor(edge_index, indices, user_mapping, movie_mapping):
    num_users, num_movies = len(user_mapping), len(movie_mapping)

    sub_edge_index = edge_index[:, indices]
    sparse_edge_index = SparseTensor(row=torch.cat([sub_edge_index[0],sub_edge_index[1]]), 
                                    col=torch.cat([sub_edge_index[1],sub_edge_index[0]]), 
                                    sparse_sizes=(num_users + num_movies, num_users + num_movies))
    return sparse_edge_index

def get_initial_sparse_edge_indexes(edge_index, train_indices, val_indices, test_indices, user_mapping, movie_mapping):
    train_sparse_edge_index = get_sparse_tensor(edge_index, train_indices, user_mapping, movie_mapping)
    val_sparse_edge_index = get_sparse_tensor(edge_index, val_indices, user_mapping, movie_mapping)
    test_sparse_edge_index = get_sparse_tensor(edge_index, test_indices, user_mapping, movie_mapping)

    return train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index    
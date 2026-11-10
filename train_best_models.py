import torch
import pandas as pd
import logging
from experiment_functions import *
from helper_functions import *
import os

experiments = [f'historic_results/e_100K/pass_8/e_100K.parquet',
               f'historic_results/e_1M/pass_3/e_1M.parquet',
               f'historic_results/e_gowalla/pass_0/e_gowalla.parquet',
               f'historic_results/e_yelp/pass_0/e_yelp.parquet',
               f'historic_results/e_amazon/pass_0/e_amazon.parquet']

dfs = []
evals_at_epochs = []

for experiment in experiments:
    df = pd.read_parquet(experiment)
    df = df.sort_values('experiment', ascending=True)
    dfs.append(df)
    eval_at_epochs = df['epochs_per_eval'].unique()[0]
    evals_at_epochs.append(eval_at_epochs)

df = pd.concat(dfs, axis=0).reset_index(drop=True)

train_df = df[~df['experiment'].str.contains('strict')].copy()
train_df['experiment_grp'] =train_df['experiment'].map(lambda x: '_'.join([x.split(sep='_')[0],x.split(sep='_')[1]]))
train_df['experiment'] = train_df['experiment'].map(lambda x: x[:-1] if x[-1] in ['1','2','3','4','5'] else x)
train_df = train_df.sort_values(['experiment','best_ndcg'],ascending=[True,False])
row_numbers = train_df.groupby('experiment').cumcount()
train_df = train_df[row_numbers.values==0]
train_df = train_df.set_index("experiment")


parameters = {
    "ITERATIONS": 4000,
    "ITERS_PER_EVAL": 50,
    "ITERS_PER_LR_DECAY": 50,
    "K": 20,
    "n_dominance": 3,
    "random_runs": 5,
    "verbose": False,
    "rating_threshold": 0,
    "patience": 10,
    "multi_objective": True,
    "objective_metric": "ndcg",
    "store_path": "historic_results/all/",
    "file_path": "train_results.parquet"
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{parameters['store_path']}all.log", mode="w"),
        logging.StreamHandler() # Also print to console
    ]
)


for exp_name in train_df['experiment_grp'].unique():
    set_seed(42)

    logger.info(f"Processing dataset: {exp_name}")
    user_mapping, movie_mapping, edge_index = load_data_generic(exp_name)
    logger.info('**** Generating the graph:')
    graph = generate_graph(edge_index)

    train_indices, val_indices, test_indices, train_edge_index, val_edge_index, test_edge_index = (
        split_data_stratified(edge_index)
    )
    train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index = (
        get_initial_sparse_edge_indexes(
            edge_index,
            train_indices,
            val_indices,
            test_indices,
            user_mapping,
            movie_mapping,
        )
    )
        
    # --- 1. Full Dataset ---
    stats = train_and_save(
        train_edge_index, train_sparse_edge_index, f"{exp_name}_full"
    )

    # --- 2. Dominant Sets & Best Random Run ---
    for m in range(1, parameters["n_dominance"] + 1):
        exp_base = f"{exp_name}_1c{m}dcs"
        logger.info(f'**** Training {exp_base}')
        logger.info(f'  ** {m}-Dominant stats:')
        logger.info('  ** Finding the dominant set:')
        dom_set = dominating_set_fast(graph, m, optimize=True)

        for strict in [False]:
            suffix = "_strict" if strict else ""
            current_exp_name = exp_base + suffix

            logger.info('  ** Getting the trainable edges:')

            ds_indices = get_ds_edges(dom_set, edge_index, graph, strict=strict)
            ds_train_indices = list(set(train_indices) & set(ds_indices))
            ds_train_edge_index = edge_index[:, ds_train_indices]

            logger.info(f'  **** Getting {len(ds_indices)} dominant set edges')
            ds_train_sparse_edge_index = get_sparse_tensor(
                edge_index, ds_train_indices, user_mapping, movie_mapping
            )

            # Train & save Dominant Set model
            train_and_save(
                ds_train_edge_index, ds_train_sparse_edge_index, current_exp_name
            )

            # Train & save Best Random Run model (using the best random run name stored in your table)
            best_rand_name = f"{current_exp_name}_rand"  # or specific random index
            if best_rand_name in train_df.index:
                set_seed(42)  # Ensure identical random sample selection
                rnd_train_indices = random.sample(train_indices, len(ds_train_indices))
                rnd_train_edge_index = edge_index[:, rnd_train_indices]
                rnd_train_sparse_edge_index = get_sparse_tensor(
                    edge_index, rnd_train_indices, user_mapping, movie_mapping
                )
                train_and_save(
                    rnd_train_edge_index, rnd_train_sparse_edge_index, best_rand_name
                )
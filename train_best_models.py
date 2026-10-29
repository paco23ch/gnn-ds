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

def append_or_create_parquet(file_path, new_rows):
  new_rows_df = pd.DataFrame(new_rows)
  if os.path.exists(file_path):
    # Read existing parquet file
    existing_df = pd.read_parquet(file_path)
    # Combine old and new rows
    combined_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
    # Save back to parquet
    combined_df.to_parquet(file_path, engine="pyarrow", index=False)
  else:
    # Create the parquet file fresh
    new_rows_df.to_parquet(file_path, engine="pyarrow", index=False)

# Helper function to train, calculate size, and save
def train_and_save(train_ei, train_sparse_ei, current_exp_name):
    return_value = {}

    if current_exp_name not in train_df.index:
      logger.warning(f"No parameters found for {current_exp_name}, skipping.")
      return

    row = train_df.loc[current_exp_name]

    # Extract best hyperparams
    emb_dim = int(row["embedding_dim"])
    n_layers = int(row["n_layers"])
    lr = float(row["lr"])
    batch_size = int(row["batch_size"])
    lambda_val = float(row["lambda_val"])
    optimizer_name = str(row["optimizer"])

    logger.info(f'**** Initializing {current_exp_name}')    
    # Instantiate objective/model runner with input params (bypassing Optuna)
    objective = GNNObjective(
        edge_index,
        train_ei,
        train_sparse_ei,
        val_edge_index,
        val_sparse_edge_index,
        user_mapping,
        movie_mapping,
        parameters,
    )
    logger.info(f'**** Setting up the model {current_exp_name}')
    objective.setup_model(emb_dim, n_layers, lr, batch_size, lambda_val, optimizer_name)

    # 1. Calculate Model Size and Parameters
    total_params = sum(p.numel() for p in objective.model.parameters())
    trainable_params = sum( p.numel() for p in objective.model.parameters() if p.requires_grad )
    param_size = sum( p.numel() * p.element_size() for p in objective.model.parameters() )
    buffer_size = sum( b.numel() * b.element_size() for b in objective.model.buffers() )
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)

    logger.info(
        f"[{current_exp_name}] Params: {total_params:,} | Trainable params: {trainable_params:,} |Size:"
        f" {model_size_mb:.4f} MB"
    )

    # 2. Train the model to completion
    logger.info(f'**** Training the model {current_exp_name}')
    train_results = objective.train_model(verbose=False)

    # 3. Save model state dict & metadata needed for inference
    checkpoint = {
        "state_dict": objective.model.state_dict(),
        "num_users": len(user_mapping),
        "num_movies": len(movie_mapping),
        "train_edges": len(train_ei[0]),
        "embedding_dim": emb_dim,
        "n_layers": n_layers,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
    }
    torch.save(checkpoint, f"{parameters['store_path']}saved_model_{current_exp_name}.pt")
    logger.info(f"Saved checkpoint: {parameters['store_path']}saved_model_{current_exp_name}.pt")

    checkpoint.pop("state_dict")
    return_value['experiment'] = current_exp_name
    return_value.update(checkpoint)
    return_value.update(train_results) 
    append_or_create_parquet(parameters['store_path']+parameters['file_path'],[return_value])
    
    return return_value 

for exp_name in ['e_100K']: #train_df['experiment_grp'].unique():
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
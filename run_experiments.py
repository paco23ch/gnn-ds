from experiment_functions import *
from helper_functions import *
import logging
import time

# define constants
parameters = {
        'ITERATIONS' : 10000,
        'ITERS_PER_EVAL' : 200,
        'ITERS_PER_LR_DECAY' : 200,
        'K' : 20, #K value for ranking metrics
        'n_trials' : 200,
        'n_dominance': 3,
        'random_runs' : 5,
        'exp_name' : 'e_100K',
        'verbose' : False,
        'movie_path' : './ml-latest-small/movies.csv',
        'rating_path' : './ml-latest-small/ratings.csv',
        #'movie_path' : './ml-1m/movies.dat',
        #'rating_path' : './ml-1m/ratings.dat'
    }

exp_name = parameters['exp_name']
optuna_storage = f'sqlite:///{exp_name}.sqlite3'
exp = f'{exp_name}_full'

# Configure the global root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{exp_name}.log", mode="w"),
        logging.StreamHandler() # Also print to console
    ]
)

logger = logging.getLogger()

optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

logger.info(f'**** Training {exp}')

user_mapping, movie_mapping, edge_index = load_data(parameters['movie_path'], parameters['rating_path'])

logger.info('**** Generating the graph:')
graph = generate_graph(edge_index)

train_indices, val_indices, test_indices, train_edge_index, val_edge_index, test_edge_index = split_data(edge_index)
train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index = \
    get_initial_sparse_edge_indexes(edge_index, train_indices, val_indices, test_indices, user_mapping, movie_mapping)

# Run the full graph experiment
runner = ExperimentRunner(edge_index, val_edge_index, val_sparse_edge_index, test_edge_index, test_sparse_edge_index, 
                            user_mapping, movie_mapping, parameters)

runner.run_experiment(train_edge_index, train_sparse_edge_index, experiment_name=exp, storage=optuna_storage)

for m in range(1, parameters['n_dominance'] + 1):
    exp = f'{exp_name}_1c{m}dcs'

    logger.info(f'**** Training {exp}')
    logger.info(f'  ** {m}-Dominant stats:')
    logger.info('  ** Finding the dominant set:')
    start = time.perf_counter()
    dom_set = dominating_set_fast(graph,m,optimize=True) #Optimized function to use sets and go through less node
    ds_time = time.perf_counter() - start
    ds_data = get_users_items(dom_set)
    ds_data.update({'ds_time':ds_time})

    logger.info('  ** Getting the trainable edges:')
    ds_indices = get_ds_edges(dom_set, edge_index, graph)
    ds_train_indices = list(set(train_indices) & set(ds_indices))
    ds_train_edge_index = edge_index[:, ds_train_indices]

    logger.info(f'  **** Getting {len(ds_indices)} dominant set edges')
    ds_train_sparse_edge_index = get_sparse_tensor(edge_index, ds_train_indices, user_mapping, movie_mapping)

    # Run the Dominant Set experiment
    runner.run_experiment(ds_train_edge_index, ds_train_sparse_edge_index, experiment_name=exp, storage=optuna_storage, ds_data=ds_data)

    for n in range(1, parameters['random_runs'] + 1):
        exp = f'{exp_name}_1c{m}dcs_rand{n}'

        logger.info(f'  **** Training {exp}')
        rnd_train_indices = random.sample(train_indices, len(ds_train_indices))

        logger.info(f'  **** Getting {len(ds_train_indices)}/{len(train_indices)} random edges')
        rnd_train_sparse_edge_index = get_sparse_tensor(edge_index, rnd_train_indices, user_mapping, movie_mapping)
        rnd_train_edge_index = edge_index[:, rnd_train_indices]

        # Run a random experiment
        runner.run_experiment(rnd_train_edge_index, rnd_train_sparse_edge_index, experiment_name=exp, storage=optuna_storage)

df = runner.get_final_results_df()
df.to_parquet(f'{exp_name}.parquet')
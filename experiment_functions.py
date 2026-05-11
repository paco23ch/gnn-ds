import time
import torch
from torch import nn, optim, Tensor
from gcn_model import *
from dominating_set_algorithms import *
from data_handling import *
import optuna
import logging

# This creates a child logger that inherits from the root you configured above
logger = logging.getLogger(__name__)

class GNNObjective:
    def __init__(self, edge_index, train_edge_index, train_sparse_edge_index, 
                                    val_edge_index, val_sparse_edge_index, 
                                    user_mapping, movie_mapping, params):

        # Store all pre-processed data as instance attributes
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.edge_index = edge_index.to(self.device)
        self.train_edge_index = train_edge_index.to(self.device)
        self.train_sparse_edge_index = train_sparse_edge_index.to(self.device)
        self.val_edge_index = val_edge_index.to(self.device)
        self.val_sparse_edge_index = val_sparse_edge_index.to(self.device)
        self.user_mapping = user_mapping
        self.num_users = len(user_mapping)
        self.num_movies = len(movie_mapping)
        self.train_edge_index_no_offset = self.train_edge_index.clone().detach()
        self.train_edge_index_no_offset[1] = self.train_edge_index_no_offset[1] - len(user_mapping)
        self.val_edge_index_no_offset = self.val_edge_index.clone().detach()
        self.val_edge_index_no_offset[1] = self.val_edge_index_no_offset[1] - len(user_mapping)
        self.ITERATIONS = params['ITERATIONS']
        self.ITERS_PER_EVAL = params['ITERS_PER_EVAL']
        self.ITERS_PER_LR_DECAY = params['ITERS_PER_LR_DECAY']
        self.K = params['K']
        self.parameters=params
        
        #logger.info(f"Using device {self.device}.")

    def __call__(self, trial):
        # 1. Suggest Hyperparameters
        n_layers = trial.suggest_int("n_layers", 1, 4)
        emb_dim = trial.suggest_categorical("embedding_dim", [32, 64, 96, 128, 256])
        LR = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        LAMBDA = trial.suggest_float("lambda_val", 1e-6, 1e-1, log=True)
        BATCH_SIZE = trial.suggest_int("batch_size", 512, 4096)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

        # 2. Setup Model
        self.setup_model(emb_dim, n_layers, LR, BATCH_SIZE, LAMBDA, optimizer_name)
        self.train_model(trial, verbose=self.parameters['verbose'])

        # 5. Return final metric
        self.model.eval()
        _, recall, _, _ = evaluation(self.model, self.val_edge_index_no_offset, self.val_sparse_edge_index, 
                                     [self.train_edge_index_no_offset], 20, self.LAMBDA)

        torch.cuda.empty_cache()

        return recall
    def setup_model(self, emb_dim, n_layers, LR, BATCH_SIZE, LAMBDA, optimizer_name):
        self.BATCH_SIZE = BATCH_SIZE
        self.LAMBDA = LAMBDA

        self.model = LightGCN(self.num_users, self.num_movies, embedding_dim=emb_dim, K=n_layers, add_self_loops=False)
        self.model = self.model.to(self.device)
        self.optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
    
    def train_model(self, trial=None, verbose=True):
        train_losses = []
        val_losses = []

        for iter in range(self.ITERATIONS):
            self.model.train()
            # forward propagation
            users_emb_final, users_emb_0, items_emb_final, items_emb_0 = self.model.forward(self.train_sparse_edge_index)

            # mini batching
            user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(self.BATCH_SIZE, self.train_edge_index_no_offset)
            user_indices, pos_item_indices, neg_item_indices = user_indices.to(self.device), pos_item_indices.to(self.device), neg_item_indices.to(self.device)
            users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
            pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
            neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

            # loss computation
            train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                                pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, self.LAMBDA)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if iter % self.ITERS_PER_EVAL == 0:
                self.model.eval()

                val_loss, recall, precision, ndcg = evaluation( self.model, self.val_edge_index_no_offset, self.val_sparse_edge_index, 
                            [self.train_edge_index_no_offset], self.K, self.LAMBDA)
                if(verbose):
                    logger.info(f"[Iteration {iter}/{self.ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, ",
                        f"val_loss: {round(val_loss, 5)}, val_recall@{self.K}: {round(recall, 5)}, val_precision@{self.K}: ",
                        f"{round(precision, 5)}, val_ndcg@{self.K}: {round(ndcg, 5)}")
                
                if trial is not None:
                    trial.report(recall, iter)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                
                train_losses.append(train_loss.item())
                val_losses.append(val_loss)
                

            if iter % self.ITERS_PER_LR_DECAY == 0 and iter != 0:
                self.scheduler.step()

        return {'train_losses': train_losses,
                'val_losses': val_losses }

    def test_model(self, test_edge_index, test_sparse_edge_index):
        self.model.eval()

        val_loss, val_recall, val_precision, val_ndcg = evaluation(self.model, self.val_edge_index_no_offset, self.val_sparse_edge_index, 
                                     [self.train_edge_index_no_offset], self.K, self.LAMBDA)
        
        self.test_edge_index = test_edge_index.to(self.device)
        self.test_sparse_edge_index = test_sparse_edge_index.to(self.device)

        self.test_edge_index_no_offset = self.test_edge_index.clone().detach()
        self.test_edge_index_no_offset[1] = self.test_edge_index_no_offset[1] - len(self.user_mapping)

        test_loss, test_recall, test_precision, test_ndcg = evaluation(
            self.model, self.test_edge_index_no_offset, self.test_sparse_edge_index, 
                [self.train_edge_index_no_offset, self.val_edge_index_no_offset], self.K, self.LAMBDA)

        return {'val_loss':val_loss, 
                'val_recall':val_recall, 
                'val_precision':val_precision, 
                'val_ndcg':val_ndcg,
                'test_loss':test_loss, 
                'test_recall':test_recall, 
                'test_precision':test_precision, 
                'test_ndcg':test_ndcg}


class ExperimentRunner():

    def __init__(self, edge_index, val_edge_index, val_sparse_edge_index, test_edge_index, test_sparse_edge_index,
                            user_mapping, movie_mapping, parameters):
        self.edge_index = edge_index
        self.val_edge_index = val_edge_index
        self.val_sparse_edge_index = val_sparse_edge_index
        self.test_edge_index = test_edge_index
        self.test_sparse_edge_index = test_sparse_edge_index
        self.user_mapping = user_mapping
        self.movie_mapping = movie_mapping
        self.parameters = parameters
        self.final_results = {}

    def get_final_results(self):
        return self.final_results

    def get_final_results_df(self):
        dfs = []

        final = self.get_final_results()
        for k in final.keys():
            df = pd.json_normalize(final[k])
            df['experiment'] = k
            dfs.append(df)

        df = pd.concat(dfs, axis = 0).set_index('experiment').reset_index()

        return(df)

    def run_experiment(self, train_edge_index, train_sparse_edge_index, experiment_name, storage, ds_data=None):
        objective = GNNObjective(self.edge_index, train_edge_index, train_sparse_edge_index, 
                                                self.val_edge_index, self.val_sparse_edge_index, 
                                                self.user_mapping, self.movie_mapping, self.parameters)
        study = optuna.create_study(direction="maximize", study_name=f"Study_{experiment_name}", 
                                    storage=storage, load_if_exists=True, 
                                    pruner=optuna.pruners.MedianPruner(
                                        n_startup_trials=5,  # Don't prune until 5 trials are done
                                        n_warmup_steps=1000  # Don't prune until at least epoch 1000
                                    )
                                )
        
        start_time = time.perf_counter()
        study.optimize(objective, n_trials=self.parameters['n_trials'])
        optim_time = time.perf_counter() - start_time

        best_params = study.best_params

        best_model = GNNObjective(self.edge_index, train_edge_index, train_sparse_edge_index, 
                                                self.val_edge_index, self.val_sparse_edge_index, 
                                                self.user_mapping, self.movie_mapping, self.parameters)
        best_model.setup_model(emb_dim=best_params['embedding_dim'], 
                                n_layers=best_params['n_layers'],
                                LR=best_params['lr'],
                                BATCH_SIZE=best_params['batch_size'], 
                                LAMBDA=best_params['lambda_val'], 
                                optimizer_name=best_params['optimizer'])
        
        start_time = time.perf_counter()
        losses = best_model.train_model(verbose=self.parameters['verbose'])
        train_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        results = best_model.test_model(self.test_edge_index, self.test_sparse_edge_index)
        inference_time = time.perf_counter() - start_time

        results['optim_time'] = optim_time
        results['train_time'] = train_time
        results['infer_time'] = inference_time
        results['users'] = len(train_edge_index[0].unique())
        results['items'] = len(train_edge_index[1].unique())
        results['train_edges']= len(train_edge_index[1])
        results['val_edges']= len(self.val_edge_index[1])
        results['test_edges']= len(self.test_edge_index[1])

        if ds_data is not None:
            results.update(ds_data)

        results.update(best_params)
        results.update(losses)

        #for m in best_params.keys():
        #    results[m] = best_params[m]

        #for m in losses.keys():
        #    results[m] = losses[m]

        self.final_results[experiment_name] = results
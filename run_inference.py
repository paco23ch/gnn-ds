import time
import torch
import pandas as pd
from gcn_model import LightGCN
from helper_functions import load_data_generic, generate_graph, split_data_stratified, get_initial_sparse_edge_indexes
from gcn_model import evaluation
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
datasets = ['e_100K', 'e_1M', 'e_gowalla', 'e_yelp', 'e_amazon']
evaluation_results = []

for exp_name in datasets:
    user_mapping, movie_mapping, edge_index = load_data_generic(exp_name)
    train_indices, val_indices, test_indices, train_edge_index, val_edge_index, test_edge_index = split_data_stratified(edge_index)
    
    train_edge_index_no_offset = train_edge_index.to(device) # clone().detach()
    train_edge_index_no_offset[1] = train_edge_index_no_offset[1] - len(user_mapping)
    
    val_edge_index_no_offset = val_edge_index.to(device) #clone().detach()
    val_edge_index_no_offset[1] = val_edge_index_no_offset[1] - len(user_mapping)

    test_edge_index = test_edge_index.to(device)
    
    test_edge_index_no_offset = test_edge_index.clone().detach()
    test_edge_index_no_offset[1] = test_edge_index_no_offset[1] - len(user_mapping)

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

    test_sparse_edge_index = test_sparse_edge_index.to(device)

    # List of all model variants you saved for this dataset
    # (Dynamically match your saved checkpoint files)
    checkpoint_files = glob.glob(f"historic_results/all/saved_model_{exp_name}_*.pt") # + glob.glob(f"historic_results/all/saved_model_{exp_name}_full.pt")

    for ckpt_path in checkpoint_files:
        model_name = ckpt_path.replace("historic_results/all\\saved_model_", "").replace(".pt", "")
        print(f"Evaluating model: {model_name}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Re-instantiate model architecture using saved metadata
        model = LightGCN(
            num_users=checkpoint['num_users'],
            num_items=checkpoint['num_movies'],
            embedding_dim=checkpoint['embedding_dim'],
            K=checkpoint['n_layers'],
            add_self_loops=False
        ).to(device)
        
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # Measure Inference Time & Compute Test Metrics
        start_time = time.perf_counter()
        with torch.no_grad():
            test_loss, test_recall, test_precision, test_ndcg = evaluation(
                model, 
                test_edge_index_no_offset, 
                test_sparse_edge_index, # sparse test index if applicable
                [train_edge_index_no_offset, val_edge_index_no_offset], 
                k=20, 
                lambda_val=1e-4 # use your saved lambda if needed
            )
        infer_time = time.perf_counter() - start_time

        evaluation_results.append({
            'experiment': model_name,
            'num_users': checkpoint['num_users'],
            'num_items': checkpoint['num_movies'],
            'total_edges': len(edge_index[0]),
            'total_train_edges': len(train_indices), 
            'total_val_edges': len(val_indices), 
            'total_test_edges': len(test_indices),
            'total_params': checkpoint['total_params'],
            'model_size_mb': checkpoint['model_size_mb'],
            'test_loss': test_loss,
            'test_recall': test_recall,
            'test_precision': test_precision,
            'test_ndcg': test_ndcg,
            'inference_time_sec': infer_time
        })

        del model, checkpoint
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save final evaluation summary
        df_eval = pd.DataFrame(evaluation_results)
        df_eval.to_parquet("final_evaluation_metrics.parquet")

print("Evaluation complete. Results saved.")
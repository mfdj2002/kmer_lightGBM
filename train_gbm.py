import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
from glob import glob
import time
import os
from tqdm import tqdm

def train_lightgbm_selective(feature_files, labels_npz, row_indices_npy, val_fraction=0.2, seed=42, **gbm_params):
    """
    Train LightGBM model with row-based validation split
    
    Args:
        feature_files: List of paths to feature matrix .npz files
        labels_npz: Path to the full label matrix .npz file
        row_indices_npy: Path to .npy file containing global row indices to include
        val_fraction: Fraction of rows to use for validation
        seed: Random seed for validation split
        **gbm_params: Additional LightGBM parameters
    """
    # Load full label matrix and row indices
    print("Loading label matrix and row indices...")
    labels = sp.load_npz(labels_npz).toarray().astype(np.int8)
    selected_indices = np.load(row_indices_npy)
    
    # Split indices into train and validation
    np.random.seed(seed)
    np.random.shuffle(selected_indices)
    n_val = int(len(selected_indices) * val_fraction)
    val_indices = selected_indices[:n_val]
    train_indices = selected_indices[n_val:]
    
    # Sort indices for efficient chunk processing
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    
    print(f"Split into {len(train_indices)} training and {len(val_indices)} validation rows")
    
    params = {
        'num_threads': 124,
        'tree_learner': 'data',
        'force_row_wise': True,
        'objective': 'binary',
        'metric': 'auc', #'binary_logloss',
        'feature_pre_filter': False,
        'num_leaves': 255,
        'feature_fraction': 0.1,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 100,
        'parallel_trees': True,
        'learning_rate': 0.2,
        'num_iterations': 50,
        'max_depth': 12,
        'histogram_pool_size': -1,
        'binary_feature': True,
        'is_unbalance': True,
        'boost_from_average': False,
        'zero_as_missing': False,
        'max_bin': 2,
        'verbose': 1,
        'sparse_threshold': 0.03,
        'use_missing': True,
        'two_round': True,
    }
    
    params = {**params, **gbm_params}
    
    # Initialize lists to store features and labels
    train_features_list = []
    train_labels_list = []
    val_features_list = []
    val_labels_list = []
    global_position = 0
    
    # Process each chunk
    for i, file_path in tqdm(enumerate(feature_files), total=len(feature_files), desc="Processing chunks"):
        # Load chunk
        features = sp.load_npz(file_path)
        chunk_size = features.shape[0]
        chunk_end = global_position + chunk_size
        
        # Process training data
        train_mask = (train_indices >= global_position) & (train_indices < chunk_end)
        train_chunk_indices = train_indices[train_mask]
        if len(train_chunk_indices) > 0:
            relative_indices = train_chunk_indices - global_position
            train_features_list.append(features[relative_indices])
            train_labels_list.append(labels[train_chunk_indices])
        
        # Process validation data
        val_mask = (val_indices >= global_position) & (val_indices < chunk_end)
        val_chunk_indices = val_indices[val_mask]
        if len(val_chunk_indices) > 0:
            relative_indices = val_chunk_indices - global_position
            val_features_list.append(features[relative_indices])
            val_labels_list.append(labels[val_chunk_indices])
        
        global_position += chunk_size
        del features
    
    # Combine all features and labels
    if not train_features_list:
        raise ValueError("No valid training rows were selected!")
    
    train_features = sp.vstack(train_features_list)
    train_labels = np.concatenate(train_labels_list)
    train_set = lgb.Dataset(train_features, label=train_labels)
    
    val_set = None
    if val_features_list:
        val_features = sp.vstack(val_features_list)
        val_labels = np.concatenate(val_labels_list)
        val_set = lgb.Dataset(val_features, label=val_labels)
    
    print("\nStarting training...")
    model = lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=[train_set, val_set] if val_set else [train_set],
        valid_names=['train', 'valid'] if val_set else ['train'],
        callbacks=[
            lgb.callback.log_evaluation(period=1),
            lgb.early_stopping(stopping_rounds=10)
        ],
        num_boost_round=500
    )


    
    
    return model, train_indices, val_indices

# Example usage
if __name__ == "__main__":
    feature_files = sorted(glob('final_csr/*.npz'))[:len(glob('final_csr/*.npz'))//100]
    labels_path = 'label_csr/amr_labels.npz'
    row_indices_path = 'label_csr/selected_rows.npy'
    
    model, train_idx, val_idx = train_lightgbm_selective(
        feature_files=feature_files,
        labels_npz=labels_path,
        row_indices_npy=row_indices_path,
        val_fraction=0.2  # 20% of rows for validation
    )
    
    # Save model and indices
    model.save_model('trained_model.txt')
    np.save('train_indices.npy', train_idx)
    np.save('val_indices.npy', val_idx)
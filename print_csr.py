import scipy.sparse as sp
import numpy as np

def print_sparse_matrix(filename):
    # Load the matrix
    matrix = sp.load_npz(filename)
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Number of non-zero elements: {matrix.nnz}")
    print(f"Sparsity: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")
    
    # Check if matrix is binary
    is_binary = is_binary_matrix(matrix)
    print(f"Is binary matrix: {is_binary}")
    
    # For small matrices, print the full dense array
    if matrix.shape[0] * matrix.shape[1] < 1000:  # Only print if matrix is small
        print("\nFull matrix:")
        print(matrix.toarray())
    else:
        print("\nFirst 5x5 corner of matrix:")
        dense_corner = matrix[:10, :10].toarray()
        print(dense_corner)

def is_binary_matrix(matrix):
    """
    Check if a sparse matrix contains only 0s and 1s.
    
    Parameters:
    -----------
    matrix : scipy.sparse matrix
        Input sparse matrix to check
    
    Returns:
    --------
    bool
        True if matrix contains only 0s and 1s, False otherwise
    """
    # Get the data array of non-zero elements
    data = matrix.data
    
    # Check if all elements are either 0 or 1
    # We only need to check non-zero elements since sparse matrices store zeros implicitly
    return np.all(np.logical_or(data == 0, data == 1))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py matrix.npz")
    else:
        print_sparse_matrix(sys.argv[1])
from pathlib import Path
import scipy.sparse as sp
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import sys

def process_single_file(fasta, kmer_to_idx):
    """Process a single file and return its data and indices"""
    col_indices = []
    with open(fasta) as f:
        while True:
            value_line = f.readline().strip()
            if not value_line:
                break
            kmer = f.readline().strip()
            if not kmer:
                break
            if kmer in kmer_to_idx:
                col_indices.append(kmer_to_idx[kmer])
    
    data = np.ones(len(col_indices))
    return data, col_indices

def parallel_second_pass(fasta_files, kmer_to_idx, max_workers=256, batch_size=256*2):
    """Process files in parallel in batches"""
    indptr = [0]
    indices = []
    data = []
    
    # Process files in batches
    for i in range(0, len(fasta_files), batch_size):
        batch_files = fasta_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{len(fasta_files)//batch_size + 1}, starting time: {datetime.now()}", flush=True)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for fasta in batch_files:
                futures.append(executor.submit(process_single_file, fasta, kmer_to_idx))
            
            # completed = 0
            for future in futures:
                row_data, row_indices = future.result()
                indices.extend(row_indices)
                data.extend(row_data)
                indptr.append(len(indices))
                # completed += 1
                # if completed % 50 == 0:
                #     print(f"Completed {completed}/{len(batch_files)} files in current batch, time now: {datetime.now()}", flush=True)
    
    return data, indices, indptr

def main(n):
    # Main code stays the same
    print("Loading kmers from file...")
    kmer_dict = pickle.load(open('../kmer_dict.pkl', 'rb'))


    fasta_files = sorted(Path('/home/kite/sparse_dump').glob('*.fasta'))
    start_idx = n * len(fasta_files) // 100
    end_idx = (n + 1) * len(fasta_files) // 100
    fasta_files = fasta_files[start_idx:end_idx]
    print(f"Processing {len(fasta_files)} files, start index: {start_idx}, end index: {end_idx}", flush=True)
    # assert len(fasta_files) == 23864

    print("Processing files in parallel...")
    data, indices, indptr = parallel_second_pass(fasta_files, kmer_dict)

    print("Creating CSR matrix...", flush=True)
    matrix = sp.csr_matrix((data, indices, indptr), 
                        shape=(len(fasta_files), len(kmer_dict)))

    print("Saving matrix... time now: ", datetime.now(), flush=True)
    sp.save_npz(f'../final_csr/sparse_matrix_{n}.npz', matrix)
    print("Done! time now: ", datetime.now(), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
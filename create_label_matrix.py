import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path

def create_amr_label_matrix(metadata_path, genome_files_list_path, quality_csv_path):
    """
    Create a sparse label matrix for antimicrobial resistance data
    
    Args:
        metadata_path: Path to BVBRC_genome_amr.csv
        genome_files_list_path: Path to txt file containing list of genome file paths
        quality_csv_path: Path to enterobacterales_complete_or_WGS_good_quality.csv
    
    Returns:
        labels_csr: CSR matrix of shape (n_genomes, n_antibiotics)
        genome_ids: List of genome IDs in the order they appear in the matrix
        antibiotic_names: List of antibiotic names in the order they appear in columns
    """
    # Load metadata and quality data
    metadata = pd.read_csv(metadata_path, dtype={'Genome ID': str})
    quality_data = pd.read_csv(quality_csv_path, dtype={'Genome ID': str})
    
    # Get intersection of genome IDs
    metadata_uids = metadata['Genome ID'].unique()
    quality_uids = quality_data['Genome ID'].unique()

    # print(f"metadata_uids: {metadata_uids[:10]}")
    # print(f"quality_uids: {quality_uids[:10]}")
    valid_genome_ids = np.intersect1d(metadata_uids, quality_uids)
    # print(f"sample of valid genome ids: {valid_genome_ids[:10]}")
    
    # Create antibiotic mapping
    unique_antibiotics = metadata['Antibiotic'].unique()
    antibiotic_to_idx = {ab: idx for idx, ab in enumerate(unique_antibiotics)}
    
    # Read list of genome files
    with open(genome_files_list_path, 'r') as f:
        genome_files = [line.strip() for line in f]
    
    # Initialize lists for CSR construction
    row_indices = []
    col_indices = []
    data = []
    included_rows = []
    current_row = 0
    
    # Process each genome file path
    for genome_file in genome_files:
        # print(f"Processing {genome_file}")
        # Extract genome ID from filename
        genome_id = Path(genome_file).stem.split('_')[1]  # Gets "108619.101" from "sparse_108619.101.fasta"
        
        # print(f"Genome ID: {genome_id}, {type(genome_id)}, str: {str(genome_id)}")
        genome_id = str(genome_id)
        # Check if genome is in valid set
        if genome_id in valid_genome_ids:
            # Get all rows for this genome in metadata
            genome_data = metadata[metadata['Genome ID'] == genome_id]
            
            # Process each antibiotic result for this genome
            for _, row in genome_data.iterrows():
                antibiotic = row['Antibiotic']
                phenotype = row['Resistant Phenotype']
                
                # Only process if it's a clear resistant/susceptible case
                if phenotype == 'Resistant':
                    row_indices.append(current_row)
                    col_indices.append(antibiotic_to_idx[antibiotic])
                    data.append(1)
                elif phenotype == 'Susceptible':
                    row_indices.append(current_row)
                    col_indices.append(antibiotic_to_idx[antibiotic])
                    data.append(0)
                # Ignore other phenotypes (intermediate, etc.)
            
            included_rows.append(current_row)
            current_row += 1
    
    # Create the sparse matrix
    labels_csr = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(included_rows), len(antibiotic_to_idx))
    )
    
    return labels_csr, included_rows, list(antibiotic_to_idx.keys())

# Example usage
if __name__ == "__main__":
    metadata_path = "BVBRC_genome_amr.csv"
    genome_files_list_path = "fasta_paths.txt"  # Your text file with the file paths
    quality_csv_path = "enterobacterales_complete_or_WGS_good_quality.csv"
    
    labels_csr, selected_rows, antibiotics = create_amr_label_matrix(
        metadata_path,
        genome_files_list_path,
        quality_csv_path
    )
    
    print(f"Label matrix shape: {labels_csr.shape}")
    print(f"Number of genomes included: {len(selected_rows)}")
    print(f"Number of antibiotics: {len(antibiotics)}")
    print("\nSample of included genome IDs:")
    print(selected_rows[:5])
    print("\nAll antibiotics:")
    for idx, ab in enumerate(antibiotics):
        print(f"{idx}: {ab}")
    
    # Save the results
    np.save('label_csr/selected_rows.npy', np.array(selected_rows))
    np.save('label_csr/antibiotics.npy', np.array(antibiotics))
    from scipy.sparse import save_npz
    save_npz('label_csr/amr_labels.npz', labels_csr)
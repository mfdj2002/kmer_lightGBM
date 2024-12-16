import pickle

print("Loading kmers from file...")
kmer_dict = {}
with open('ALL.fasta', 'r') as f:
    idx = 0
    while True:
        header = f.readline().strip()  # Read >number line
        if not header:  # EOF
            break
        kmer = f.readline().strip()    # Read kmer line
        if not kmer:    # EOF
            break
        kmer_dict[kmer] = idx
        idx += 1

# Save dictionary
with open('kmer_dict.pkl', 'wb') as f:
    pickle.dump(kmer_dict, f)

print(f"done building kmer_dict, {len(kmer_dict)} kmers")
To reproduce data preprocessing:

run in the following order

python jellyfish/run_jf_counts.py #which does jellyfish count and creates <kmers for each sequence>.jf
python jellyfish/merge_all.py #which creates an ALL.jf
python jellyfish/dump_kmers.py #which dumps the <kmers for each sequence>.jf into .fasta files
(you also need to manually dump the ALL.jf into ALL.fasta)

python create_pkl.py #constructs the hashset of all kmers that has appeared among all sequences
python create_csr.py #creates the sparse matrix where rows are genomes and columns are kmers
python create_label_matrix.py #creates the sparse matrix for labels, rows are again genomes and columns are antibiotics

reference lightgbm implementation:
python train_gbm.py

reference implementation for evo-bert pipeline:
after properly setting up evo, replace its model.py with evo/model.py #only one line change, forces it to output the hidden representation in last layer in the last position for each inference step

python evo/collect_representation.py
first collect the hidden representation in last layer in the last position that Evo outputs from the input which is either of length 10000 or one whole contig
This representation is basically "what Evo interprets, given the input chunk"

python train_bert.py
train bert on the collected representations. This is done as a way to compress the long context needed for processing whole genomes. Even if Evo is known for its long context, it's only trained on 131k context length, but a bacteria on average has 3-5 million base pairs, human has 3billion. 



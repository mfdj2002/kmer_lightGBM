from evo import Evo
import torch
import time
import glob


device = 'cuda:0'

evo_model = Evo('evo-1-131k-base')
model, tokenizer = evo_model.model, evo_model.tokenizer
model.to(device)
model.eval()

def process_sequence_chunk(chunk, genome_idx, genome_chunk_idx):
    
    sequence = chunk
    input_ids = torch.tensor(
        tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).to(device).unsqueeze(0)

    with torch.no_grad():
        _, _, last_hidden_state_last_pos = model(input_ids, genome_idx=genome_idx, genome_chunk_idx=genome_chunk_idx) # (batch, length, vocab)

    return last_hidden_state_last_pos

fasta_files = glob.glob('fasta_files/*.fasta')


for i, fasta_file in enumerate(fasta_files):
    try:
        with open(fasta_file, 'r') as file:
            chunk = ""
            current_representation = []
            sequence_length = 0
            j = 0
            
            for line in file:
                # If we find a new sequence header
                if line.startswith('>'):
                    # Process previous chunk if it exists
                    if chunk:
                        current_representation.append(process_sequence_chunk(chunk, i, j))
                        chunk = ""
                        sequence_length = 0
                    
                    j+=1
                    continue
                
                # Add sequence line to current chunk
                sequence_line = line.strip()
                new_length = sequence_length + len(sequence_line)
                
                # Check if adding this line would exceed token limit
                if new_length >= 10000:
                    # Process current chunk before it exceeds limit
                    current_representation.append(process_sequence_chunk(chunk, i, j))
                    # Start new chunk with current line
                    chunk = sequence_line
                    sequence_length = len(sequence_line)
                    j+=1
                else:
                    # Add to current chunk
                    chunk += sequence_line
                    sequence_length = new_length
                
            
            # Process final chunk if it exists
            if chunk:
                current_representation.append(process_sequence_chunk(chunk, i, j))
            
            current_representation = torch.cat(current_representation, dim=0)
            torch.save(current_representation, f"representations/{fasta_file.split('/')[-1]}.pt")
                
    except Exception as e:
        print(f"Error processing file {fasta_file}: {str(e)}")


import os
import subprocess
from tqdm import tqdm

# Directories
input_directory = "/home/genomes/genomes"
output_directory = "/home/Shawn_xgb/Jellyfish"

# Jellyfish command parameters
jellyfish_command = "jellyfish"
kmer_length = 15
hash_size = "500M"
threads = 8

def run_jellyfish():
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # List all files in the input directory
    fasta_files = [f for f in os.listdir(input_directory) if f.endswith(".fasta")]

    # Ensure there are FASTA files to process
    if not fasta_files:
        print("No FASTA files found in the directory!")
        return

    # Initialize the progress bar
    with tqdm(total=len(fasta_files), desc="Processing Files", unit="file") as progress_bar:
        # Process each FASTA file
        for fasta_file in fasta_files:
            # Define input and output paths
            input_path = os.path.join(input_directory, fasta_file)
            output_path = os.path.join(output_directory, os.path.splitext(fasta_file)[0] + ".jf")

            # Check if the corresponding .jf file already exists
            if os.path.exists(output_path):
                progress_bar.update(1)  # Update the progress bar for skipped files
                continue

            # Construct the Jellyfish command
            command = [
                jellyfish_command,
                "count",
                "-m", str(kmer_length),
                "-s", hash_size,
                "-t", str(threads),
                "-o", output_path,
                "-C",
                input_path
            ]

            # Run the command and capture output
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            except subprocess.CalledProcessError as e:
                print(f"Error while processing {fasta_file}: {e}")

            # Update the progress bar
            progress_bar.update(1)


if __name__ == "__main__":
    run_jellyfish()

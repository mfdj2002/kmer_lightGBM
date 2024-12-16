import os
import random
import subprocess
import sys

# Directory containing .jf files
input_directory = "/home/Shawn_xgb/Jellyfish"

# Output directory for dumped files
output_directory = "/home/Shawn_xgb/dump"

# Jellyfish command
jellyfish_command = "jellyfish"

def jf_dump(n):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # List all .jf files in the directory
    jf_files = [f for f in os.listdir(input_directory) if f.endswith(".jf")]
    start_idx = n * len(jf_files) // 8
    end_idx = (n + 1) * len(jf_files) // 8

    for jf_file in jf_files:
        input_path = os.path.join(input_directory, jf_file)
        output_file = os.path.splitext(jf_file)[0] + ".fasta"
        output_path = os.path.join(output_directory, output_file)

        # Construct the Jellyfish dump command
        command = [
            jellyfish_command,
            "dump",
            input_path,
            "-o",
            output_path
        ]

        try:
            # Run the command
            print(f"Dumping {jf_file} to {output_file}...")
            subprocess.run(command, check=True)
            print(f"Successfully dumped to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error dumping {jf_file}: {e}")

if __name__ == "__main__":
    jf_dump(int(sys.argv[1]))
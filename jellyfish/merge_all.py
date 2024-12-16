import os
import subprocess

# Directories
fasta_directory = "/home/Shawn_xgb/raw_fastas"
jf_directory = "/home/Shawn_xgb/Jellyfish"
output_merged_file = "/home/Shawn_xgb/Jellyfish/merged.jf"

# Jellyfish command
jellyfish_command = "jellyfish"

def merge_jf_from_txts():
    # List all .txt files in the directory
    fasta_files = [f for f in os.listdir(fasta_directory) if f.endswith(".fasta")]

    # Extract the base names (without extensions) of the .txt files
    base_names = [os.path.splitext(txt)[0] for txt in fasta_files]

    # Find matching .jf files
    jf_files_to_merge = [
        os.path.join(jf_directory, base_name + ".jf")
        for base_name in base_names
        if os.path.exists(os.path.join(jf_directory, base_name + ".jf"))
    ]

    # Ensure there are .jf files to merge
    if not jf_files_to_merge:
        print("No matching .jf files found to merge.")
        return

    print(f"Found {len(jf_files_to_merge)} .jf files to merge.")

    # Construct the jellyfish merge command
    command = [jellyfish_command, "merge", "-o", output_merged_file, "-m"] + jf_files_to_merge

    # Run the merge command
    try:
        print("Merging .jf files...")
        subprocess.run(command, check=True)
        print(f"Merge completed successfully. Merged file saved as {output_merged_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during merge: {e}")

if __name__ == "__main__":
    merge_jf_from_txts()

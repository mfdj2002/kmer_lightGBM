import os
import subprocess

# Directories
jf_directory = "/home/Shawn_xgb/Jellyfish"
output_merged_file = "/home/Shawn_xgb/Jellyfish/ALL.jf"

# Jellyfish command
jellyfish_command = "jellyfish"

def merge_jf():

    # Find matching .jf files
    jf_files_to_merge = os.listdir(jf_directory)

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
    merge_jf()
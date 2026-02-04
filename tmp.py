import gzip
import json
import sys
import os

def convert_jgz_to_json(input_path, output_path=None):
    """
    Converts a compressed .jgz file to a readable .json file.
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found -> {input_path}")
        return

    # If no output path is provided, simply change extension to .json
    if output_path is None:
        output_path = input_path.replace('.jgz', '.json')
        # Handle case where file didn't end in .jgz
        if output_path == input_path: 
            output_path += ".json"

    print(f"Reading {input_path}...")
    
    try:
        # 1. Open the gzip file in text read mode ('rt')
        with gzip.open(input_path, 'rt', encoding='utf-8') as f_in:
            data = json.load(f_in)

        # 2. Write to standard json file with indentation for readability
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=4)
            
        print(f"Success! Saved readable file to: {output_path}")

    except Exception as e:
        print(f"Failed to convert: {e}")

if __name__ == "__main__":
    # Check if user provided a file argument
    if len(sys.argv) < 2:
        print("Usage: python jgz2json.py <path_to_file.jgz>")
    else:
        convert_jgz_to_json(sys.argv[1])
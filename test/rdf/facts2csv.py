
import os
import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python facts2csv.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

        # list all files in the directory
    for input_file_path in os.listdir(input_dir):
        if not input_file_path.endswith(".facts"):
            continue
        rel = input_file_path.split(".")[0]
        with open(os.path.join(input_dir, input_file_path), "r") as input_file:
            with open(os.path.join(output_dir, f"{rel}.csv"), "w+") as f:
                lines = input_file.readlines()
                for line in lines:
                    parts = line.split("\t")
                    f.write(", ".join(parts))


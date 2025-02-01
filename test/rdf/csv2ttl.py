
import sys
import os


name_list = [
    "cfg_edge.facts",
    "universal_region.facts",
    "var_used_at.facts",
    "child_path.facts",
    "path_moved_at_base.facts",
    "path_assigned_at_base.facts",
    "path_accessed_at_base.facts",
    "path_is_var.facts",
]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python facts2ttl.py <input_dir> <output.ttl>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]

    with open(output_file, "w+") as f:
        f.write(f"@prefix : <https://rdfox.com/getting-started/> .\n")
        # list all files in the directory
        for input_file_path in os.listdir(input_dir):
            if not input_file_path.endswith(".facts"):
                continue
            if input_file_path not in name_list:
                continue
            
            rel = input_file_path.split(".")[0]
            with open(os.path.join(input_dir, input_file_path), "r") as input_file:
                lines = input_file.readlines()
                for line in lines:
                    parts = line.split("\t")
                    if len(parts) == 2:
                        f.write(f"{parts[0].strip()} :{rel} {parts[1].strip()} .\n")
                    elif len(parts) == 1:
                        f.write(f"{parts[0].strip()} :{rel} {parts[0].strip()} .\n")
                    else:
                        print(f"Invalid line: {len(parts)}")

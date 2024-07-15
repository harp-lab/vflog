
# extract all string from csv file under a directory

import os


def extract_string(file_path, string_set: set):
    # if end with .csv
    if file_path.endswith('.csv'):
        # open file
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                words = line.split(',')
                for word in words:
                    word = word.strip()
                    # if there is "" aroud the word remove it
                    if word.startswith('"') and word.endswith('"'):
                        word = word[1:-1]
                    if not word.isdigit():
                        string_set.add(word)


def extract_string_from_dir(dir_path):
    string_set = set()
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            extract_string(file_path, string_set)
    return string_set


def replace_string_with_id(input_dir, output_dir):
    # extract all string from input_dir
    string_set = extract_string_from_dir(input_dir)
    print(f"Extracted {len(string_set)} strings")
    # create map from string to id
    string_map = {}
    cnt = 0
    for string in string_set:
        string_map[string] = cnt
        cnt += 1
    # write map to string.tsv file under output_dir
    with open(os.path.join(output_dir, 'string.tsv'), 'w') as file:
        for string in string_map:
            file.write(f"{string_map[string]}\t{string}\n")
    # replace string with id
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            # if end with .csv
            if file_path.endswith('.csv'):
                # open file
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    new_lines = []
                    for line in lines:
                        words = line.split(',')
                        new_words = []
                        for word in words:
                            word = word.strip()
                            # if there is "" aroud the word remove it
                            if word.startswith('"') and word.endswith('"'):
                                word = word[1:-1]
                            if not word.isdigit():
                                new_words.append(str(string_map[word]))
                            else:
                                new_words.append(word)
                        new_line = '\t'.join(new_words)
                        new_lines.append(new_line)
                # write to output_dir
                # change .csv to .facts
                output_file_name = filename[:-4] + '.facts'
                output_file_path = os.path.join(output_dir, output_file_name)
                with open(output_file_path, 'w') as file:
                    for line in new_lines:
                        file.write(line + '\n')


if __name__ == '__main__':
    # 2 argunments, input dir and output dir for converted
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 extract_string.py input_dir output_dir")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_string_file = sys.argv[2]
    replace_string_with_id(input_dir, output_string_file)

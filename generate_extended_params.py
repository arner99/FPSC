"""
throwaway script for generating a parameter cldf file for the merged dataset that includes concepticon ids
"""

import os

ids = []
first_line_processed = False

with open("resources/lexibank/merged_dataset_change/cldf/parameters.csv") as f:
    for line in f:
        line = line.strip()
        if line == "" or not first_line_processed:
            first_line_processed = True
            continue
        ids.append(line.split(",")[0])

lexibank_source_dir = "resources/lexibank/source_databases_change"
databases = [name for name in os.listdir(lexibank_source_dir)
             if os.path.isdir(os.path.join(lexibank_source_dir, name))]

with open("resources/lexibank/merged_dataset_change/parameters_with_concepticon.csv", "w") as out_file:
    out_file.write("ID,Name,Concepticon_ID,Concepticon_Gloss\n")
    for db in databases:
        with open(lexibank_source_dir + "/" + db + "/cldf/parameters.csv") as f:
            for line in f:
                if '"' in line:
                    start_index = line.index('"')
                    end_index = line.index('"', start_index+1)
                    substring = line[start_index:end_index]
                    line = line[:start_index] + substring.replace(",", "PLACEHOLDER") + line[end_index:]
                relevant_content = line.strip().split(",")[:4]
                if relevant_content[0] in ids:
                    out_string = (",".join(relevant_content) + "\n").replace("PLACEHOLDER", ",")
                    out_file.write(out_string)

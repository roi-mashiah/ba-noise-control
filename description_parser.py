import pandas as pd
import json
from os import path
from glob import glob
import sys

main_dir = sys.argv[1]
jsons_dir = path.join(main_dir, "**/*.json")
print(main_dir)
all_jsons = glob(jsons_dir, recursive=True)
print("********** jsons: {}".format(len(all_jsons)))
descriptions = []
for j_path in all_jsons:
    with open(j_path, "r") as reader:
        json_desc = json.load(reader)
    folder, _ = path.split(j_path)
    json_desc["folder"] = folder
    df = pd.DataFrame().from_dict(json_desc, orient="index").T
    descriptions.append(df)
all_desc = pd.concat(descriptions, ignore_index=False)
all_desc.to_csv("descriptions.csv", index=False)
all_desc.to_excel("descriptions.xlsx", index=False)

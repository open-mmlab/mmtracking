import os
import json
from pathlib import Path
import copy
import argparse

parser = argparse.ArgumentParser(description="remap category ids in coco-format box-track label files")
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
args = parser.parse_args()

path = Path(args.input)
assert path.is_file(), f"{args.input} is not a valid path"
new_path = Path(args.output)

with open(path, 'r') as fh:
    config = json.load(fh)
    
new_categories = [{'id': 1, 'name': 'pedestrian'}, {'id': 2, 'name': 'rider'}, {'id': 3, 'name': 'car'},
                  {'id': 4, 'name': 'bus'}, {'id': 5, 'name': 'truck'}, {'id': 6, 'name': 'bicycle'},
                  {'id': 7, 'name': 'motorcycle'}, {'id': 8, 'name': 'train'}]
remap_cat_ids = {
    1: 1,
    2: 2,
    3: 3,
    4: 5,
    5: 4,
    6: 8,
    7: 7,
    8: 6
}
config["categories"] = new_categories
remapped_annotations = []
for ann in config["annotations"]:
    c_ann = copy.deepcopy(ann)
    c_ann['category_id'] = remap_cat_ids[c_ann['category_id']]
    remapped_annotations.append(c_ann)

config["annotations"] = remapped_annotations
with open(str(new_path), 'w') as fh:
    json.dump(config, fh)
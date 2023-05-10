import argparse
import pickle
import typing

import svgelements
from svgelements import *
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--dataset_path", type=str)
parser.add_argument("-sf", "--save_folder", type=str)
args = parser.parse_args()

dataset_path = args.dataset_path
save_path = args.save_folder
Path(save_path).mkdir(parents=True, exist_ok=True)
pathlist = Path(dataset_path).glob('*.svg')

svg_parser = SVG()


def to_list(svg_path: svgelements.Path):
    d = []
    for segment in svg_path.segments():
        s = str(segment.d(smooth=False))
        splitted = s.split()
        name = str(splitted[0]).upper()
        segmentArgs = d.append(name)
        for pair in splitted[1:]:
            x, y = [int(i) for i in pair.split(",")]
            d += [str(x), str(y)]
    return d


for path in pathlist:
    try:
        svg = typing.cast(SVG, svg_parser.parse(str(path)))
    except:
        print("Parse error: " + str(path))
        continue
    with open(path) as f:
        content = f.read()
        if "<!--" not in prompt:
            print("Missing prompt: " + str(path))
            prompt = None
    result = {
        "paths": [],
        "prompt": prompt
    }
    for svg_path in svg.elements(lambda e: isinstance(e, Shape)):
        result["paths"].append({
            "path": to_list(svg_path),
            "fill": str(svg_path.fill)
        })
    pickle.dump(result, open(save_path + "\\" + path.name.replace(".svg", ".pkl"), 'wb'))



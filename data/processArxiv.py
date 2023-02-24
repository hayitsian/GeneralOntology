#!/usr/bin/env python
#
# Ian Hay - 2023-02-23

import json
import pandas as pd

filepath = "arXiv/arxiv-metadata-oai-snapshot.json"
filepathCSV = "arXiv/arxiv-metadata-oai-snapshot.csv"

# data we want: "id", "categories", "abstract"

_data = []
_cols = ["id", "categories", "abstract"]

for line in open(filepath, "r"):
    arxiv_data = json.loads(line)
    arxiv_data = [arxiv_data[i] for i in _cols]
    _data.append(arxiv_data)

df = pd.DataFrame(_data, columns=_cols)
print(df.head())
print(df.shape)

df.to_csv(filepathCSV, index=False)
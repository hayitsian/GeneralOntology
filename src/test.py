#!/usr/bin/env python


import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("../visualizations/experiment 3 - 2023-03-25/Classification Metrics for Gensim LDA with Bag-of-words features on raw texts data - base labels.csv", index_col=0)
print(df.head())

print(df["perplexity"])

for _strVal in df["perplexity"].values:
    print(_strVal)
    _strValProc = _strVal.replace("[","").replace("]","").split(", ")
    _val = [float(i) for i in _strValProc]
    print(_val)


# print(pd.to_numeric(df.loc["perplexity"].str.replace("[","").str.replace("]","").str.split(", ")))
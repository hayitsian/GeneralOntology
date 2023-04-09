#!/usr/bin/env python


import pandas as pd
from model.pipeline.AXpreprocessing import AXimporter

_dataCol = "abstract"
_labelCol = "categories"
_topLabelCol = "top category"
_baseLabelCol = "base categories"
_topBaseLabelCol = "top base category"

numClasses = 8 # value is used later on
numDataPoints = 20000 # value is used later on - roughly 13,000 manuscripts per topic assuming even distribution


_rawFilename = "../data/arXiv/arxiv-metadata-oai-snapshot.csv"

importer = AXimporter()

df = importer.importData(_rawFilename,verbose=True)
df = importer.parseLabels(_labelCol, _topLabelCol, _baseLabelCol, _topBaseLabelCol, verbose=True)
dfSubsetHigherLabels = importer.getSubsetFromNClasses(df, _topBaseLabelCol, numClasses, numDataPoints, verbose=True)

print(dfSubsetHigherLabels.head())

dfSubsetHigherLabels.to_csv("../data/arXiv/arxiv-metadata-sample-8topics-20000abstracts.csv")
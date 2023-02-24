#!/usr/bin/env python
#
# Ian Hay - 2023-02-23

import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import itertools
import pandas as pd
import AXpreprocessing

#
# Make sure to run this in your environment!: python3 -m spacy download en_core_web_trf
#

def main():
    # take in filename as a command line argument
  _rawFilename = sys.argv[1] # takes in the actual file path


  _cols = ["id", "categories", "abstract"]
  _dataCol = "abstract"
  _labelCol = "categories"
  

  # open the data file
  print("\nImporting & Preprocessing data...\n\n")

  df = pd.read_csv(_rawFilename, low_memory=False)
  print(df.head())
  df[_labelCol] = df[_labelCol].str.split(expand=False)
  _categories = df[_labelCol]
  print(_categories)


  print(set(list(itertools.chain(*_categories.values))))

main()
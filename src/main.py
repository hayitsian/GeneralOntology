#!/usr/bin/env python
#
# Ian Hay - 2023-04-08
# https://github.com/hayitsian/General-Index-Visualization

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # ---- run the program ---- # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# dependencies
import argparse
from controller import BaseController


def main():

    parser = argparse.ArgumentParser(description="Topic modeling for scientific works.")

    # add command line arguments

    # required
    parser.add_argument("action", help="The action for this program to take.",
                        choices=["query", "train", "update"])
    parser.add_argument("type", help="The data type to use for this model.",
                        choices=["AX", "GI", "PM"])
    parser.add_argument("data", help="The datasource to use for this model")
    parser.add_argument("model", help="The model type to use.",
                        choices=["LDA", "SBM", "NN"])

    # optional
    parser.add_argument("-v", "--verbose", help="Set the verbosity of the output.",
                        action="store_true")
    parser.add_argument("-s", "--save", help="Whether to save the model and to what filename.")
    parser.add_argument("-l", "--load", help="Whether to load a pretrained model and the filename to load from.")
    parser.add_argument("-p", "--preprocess", help="Whether to the data.")
    parser.add_argument("-o", "--output", help="The output for this program to make, default: JSON",
                        choices=["JSON", "CSV", "TXT"], default="JSON")


    # build appropriate controller and pass into it
    # for now, just builds a base controller
    # TODO: extend the controller and build based on input args
    args = parser.parse_args()
    cont = BaseController(args)
    out = cont.run()


main()
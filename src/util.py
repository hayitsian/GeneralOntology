# Ian Hay - 2023-02-23

import sys
import inspect
# import torch
import numpy as np
from sklearn import metrics

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def requireNotNull(input):
    if input is None: raise ValueError("Input cannot be None.")


def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)


def getTopPrediction(probs):
    return np.argmax(probs, axis=1)


def getBaseCategory(str, type="AX"):
    # TODO Abstract out datatype between : "AX", "PM", "WK", "GI"
    # (for arXiv, PubMed, Wikipedia, and General Index sources, respectively)

    if type=="AX":
        return _getBaseCategoryAX(str)
    
    else: raise ValueError(f"Invalid type argument: {type}")

def _getBaseCategoryAX(str):
    if ("." in str):
        base = str.split(".")[0]
        if (base=="astro-ph" or base=="nlin" or base=="cond-mat"): return "physics"
        else: return base
    elif str=="q-alg" or str=="alg-geom" or str=="dg-ga" or str=="funct-an": return "math"
    elif str=="q-bio": return "q-bio"
    elif str=="cmp-lg": return "cs"
    else: return "physics"


def getBaseCategories(listStr, type="AX"):
    baseList = [getBaseCategory(s, type) for s in listStr]
    _baseList = list({s:0 for s in baseList}) # ordered set: https://stackoverflow.com/questions/51145707/using-ordered-dictionary-as-ordered-set
    # return list(set(baseList)) # this is stochastic and does not preserve order of elements
    return _baseList


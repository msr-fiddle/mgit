import os
import sys
MGIT_PATH=os.path.dirname(os.getcwd())
sys.path.append(MGIT_PATH)
from utils.lineage.graph import *  # MGit repository needs to be in PYTHONPATH.
from utils.lcs.diffcheck import diff_lcs
from utils.ht.diffcheck import diff_ht
from utils.model_utils import load_models
from utils import meta_functions
import pdb
import subprocess
import traceback

"""
Extract the model name (ex: bert-large-cased) from its full path.
"""


def extractModelNameFromPath(name):
    l = name.split("/")
    return l[len(l) - 1]


"""
Extract the model name (ex: bert-large-cased) from the string in the window list.
"""


def extractModelNameFromTree(name):
    s = name.strip()
    idx = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i] in [" ", "]"]:
            idx = i
            break
    return s[idx + 1 :]


"""
List connected models for a given model. Direction means: 0 - children models; 1 - parent models;
"""


def listConnectedModels(graph, m_type, direction, model):
    models = {*()}
    if m_type == "all":
        for k in graph.log["adapted"].keys():
            if k[direction] == model:
                models.add(k[(direction + 1) % 2])
        for k in graph.log["versioned"].keys():
            if k[direction] == model:
                models.add(k[(direction + 1) % 2])
    else:
        for k in graph.log[m_type].keys():
            if k[direction] == model:
                models.add(k[(direction + 1) % 2])
    models_list = []
    for k in models:
        models_list.append(k)
    return models_list


"""
Create a list of all children models from a given model in the graph. This list
corresponds to the depth first travesal result. Also, each element of the list
contains the indentation of each child node, in order to build the tree view shown
in the window. The <seen> variable guarantees the same node is not traversed more
than one.
"""


def treeConnectedModels(graph, m_type, direction, model):
    seen = {}
    return treeConnectedModelsRecursive(graph, m_type, direction, model, 0, seen), seen


"""
Recursive implementation of treeConnectedModels.
"""


def treeConnectedModelsRecursive(graph, m_type, direction, model, level, seen):
    if not model in seen:
        s = len(seen)
        seen[model] = s + 1
        models = listConnectedModels(graph, m_type, direction, model)
        if len(models) > 0:
            children = []
            idx = 0
            for m in models:
                if m != "root":
                    # [(level,m,last)]: level -> indentation, m -> model, last -> if the model is the last child
                    if idx == len(models) - 1:
                        children += [(level, m, True)] + treeConnectedModelsRecursive(
                            graph, m_type, direction, m, level + 2, seen
                        )
                    else:
                        children += [(level, m, False)] + treeConnectedModelsRecursive(
                            graph, m_type, direction, m, level + 2, seen
                        )
                    idx += 1
            return children
        return []
    return []


"""
Load a text file.
"""


def loadFile(filename):
    with open(filename) as m_file:
        lines = m_file.readlines()
    return lines


"""
Create a list of strings representing the tree view with indentation.
parameters:
    nodes: List of nodes.
    modelsIdx: If present, use as the mapping node -> idx in the main
    window list, if not build this map and return it.
"""


def createTreeView(nodes, modelsIdx={}):
    items = []
    indent = {}
    seenModelsIdx = {}
    modelToTreeIdx = {}
    idx = -1
    newIdx = 0
    # the <indent> variable is created as a list with one element per indentation,
    # and each of these elements is a list with the strings already created with
    # the corresponding indentation, filled with spaces on left
    for value in nodes:
        shortName = extractModelNameFromPath(value[1])
        idx += 1
        if shortName in modelsIdx:
            idxAux = modelsIdx[shortName]
        else:
            if shortName in seenModelsIdx:
                idxAux = seenModelsIdx[shortName]
            else:
                idxAux = newIdx
                newIdx += 1

        if not idxAux in modelToTreeIdx:
            modelToTreeIdx[idxAux] = idx

        if value[0] in indent:
            indent[value[0]] += [(idx, value[2])]
        else:
            indent[value[0]] = [(idx, value[2])]

        if shortName in seenModelsIdx:
            items.append(list(" " * (value[0] + 1) + " [%d+] " % (idxAux) + shortName))
        else:
            items.append(list(" " * (value[0] + 1) + " [%d] " % (idxAux) + shortName))
            seenModelsIdx[shortName] = idxAux

    # Parse the <indent> list including the corresponding tree ASCII characteres on left
    for k, v in indent.items():
        idx = 0
        for e, i in enumerate(v):
            if i[1] == True:
                # it is the last item with the current indentation
                items[i[0]][k] = "└"
            else:
                items[i[0]][k] = "├"
                for j in range(i[0] + 1, v[e + 1][0]):
                    items[j][k] = "|"
            idx += 1

    # convert the arrays of chars in strings
    sItems = []
    for i in items:
        sItems.append("".join(i))
    return sItems, seenModelsIdx, modelToTreeIdx

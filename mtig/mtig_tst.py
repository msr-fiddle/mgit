import os
import sys
MGIT_PATH=os.path.dirname(os.getcwd())
sys.path.append(MGIT_PATH)
from utils.lineage.graph import *  # MGit repository needs to be in PYTHONPATH.
from utils import meta_functions
import argparse
import pdb
from mtig import *


def main():
    filename = sys.argv[1]
    g = LineageGraph.load_from_file("./", filename=filename)
    g.show(layout=False)
    pdb.set_trace()
    children = treeConnectedModels(g, "all", 0, "root")


if __name__ == "__main__":
    main()

import os
import sys
MGIT_PATH=os.path.dirname(os.getcwd())
sys.path.append(MGIT_PATH)
from utils.lineage.graph import *  # MGit repository needs to be in PYTHONPATH.
import argparse
import pdb


def main():
    pdb.set_trace()
    g = LineageGraph.load_from_file("./", load_tests=False)
    print(len(g.nodes))  # Print number of nodes in graph.
    print(g.nodes)  # Print nodes dict itself.
    print(g.log.keys())  # Get edge types.
    print(g.log["versioned"].keys())  # Get versioned edges.
    print(g.log["adapted"].keys())  # Get adapted edges.


if __name__ == "__main__":
    main()

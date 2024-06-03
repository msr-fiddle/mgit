# MGit: A Model Versioning and Management System

This repository contains the source code implementation of "[MGit: A Model Versioning and Management System](https://arxiv.org/abs/2307.07507)" (to appear at ICML 2024).
This source code is available under the [MIT License](LICENSE.txt).

More documentation coming soon!

## Directory Structure

### `utils`

`utils` contains the implementation of the MGit API to construct, mutate and save the lineage graph.

### `notebooks`

This sub-directory Jupyter notebooks demonstrating usage of MGit's API, and also the various experiments shown in the paper.

## Setup

### Software Dependencies

To run the software in this repository, you will need to pull the [NVIDIA PyTorch 21.12 docker image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-12.html#rel_21.12) or equivalent. Please run ```bash setup.sh``` to fix the python hash seed and also install
all other required Python dependencies (specified in `requirements.txt`).


### Data

#### Lineage graph dataset in the Paper

```scripts``` and ```notebooks/experiments/creation``` contain the code to create the models and corresponding lineage graphs.


## Python Interface
More detailed end-to-end demonstration can be found under ```notebooks/examples``` (e.g. ```TestLineageGraph``` shows how to construct G1 in the paper).

### Initialize a Lineage Graph

```python
from utils.lineage.graph import *
g = LineageGraph(compression_mode='lzma') #This sepcifies the delta compression algorithm
```

### Add a Lineage Test

```python
mlm_lineage_eval_dataset = LineageDataset('wikitext','wikitext-103-raw-v1',split='validation',feature_keys=['text'])
preprocess_file = 'utils/preprocess_utils.py'
mlm_test = LineageTest(
        preprocess_function_path=preprocess_file,
        preprocess_function_name='mlm_preprocess_function',
        eval_dataset=mlm_lineage_eval_dataset,
        metric_for_best_model='loss',
        name='mlm',
)
g.register_test_to_type(mlm_test,'mlm')
```

### Add a Lineage Node (Autopilot Mode)

```python
if not g.add(node) :
    g.add_root(node)
node = LineageNode(output_dir='models/bert-base-cased', init_checkpoint='bert-base-cased', \
                               model_type='mlm', task_type='MaskedLM', is_delta=True)
```

### Save the Lineage Graph

```python
n = g.show(save_path="./LineageGraph.html")
g.save('./')
```

## Command Line Interface

### `diff`

To run `mgit diff`, use the following:
```bash
mgit diff -c -s -o path_to_output_graph path_to_model1 path_to_model2
```

All arguments can be found by using the `-h` command line argument:
```bash
usage: mgit [-h] [-c] [-s] [-o O] [-lcs] {diff,merge,pull,push,rebase} arguments [arguments ...]

Git for (Foundation) Models.

positional arguments:
  {diff,merge,pull,push,rebase}
                        Specify the type of mgit command.
  arguments             Arguments to mgit command

optional arguments:
  -h, --help            show this help message and exit
  -c                    Specify the granularity of traced layers
  -s                    Use structural mode for the diff if specified, default mode is contextual diff
  -o O                  Specify the output file path for the generated graph
  -lcs                  Use lcs algorithm to compute diff if specified
```

This outputs the sets of nodes and edges to be removed from and to be added to model1
to construct model2. A node is a layer whose granularity can be specified by the user
using the `-c` flag.  The finest granularity (also the default) is `torch.nn.Module`.
An edge in a model specifies data flow from one layer to the other. Together, all nodes
and edges create a directed acyclic graph (DAG) representing the model. In both graph
and terminal output, an uncolored node/edge exists in both models. A red node or edge
exists in only `model1` and a green one exists only in `model2`.


## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE.txt) license.

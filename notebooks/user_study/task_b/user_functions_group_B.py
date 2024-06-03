import numpy as np
from datasets import load_metric
import copy
import os
import sys
MGIT_PATH=os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(MGIT_PATH)
from utils.lineage.graph import *
from utils.lcs.diffcheck import lcs_one

# +
# To control logging level for various modules used in the application:
import logging
import re
import warnings
from datasets.utils.logging import disable_progress_bar

def set_dataset_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
    
    disable_progress_bar()


# +
def mlm_preprocess_function(lineage_dataset, tokenizer, **kwargs):
    feature_keys = lineage_dataset.feature_keys

    def tokenize_function(examples):
        inputs = [examples[k] for k in feature_keys]
        if "labels" in examples:
            outputs = [examples["labels"]]
        else:
            outputs = []
            for k in feature_keys:
                if "org_" + k in examples:
                    text_list = []
                    for i in range(len(examples["org_" + k])):
                        if examples["org_" + k][i] is None:
                            text_list.append(examples[k][i])
                        else:
                            text_list.append(examples["org_" + k][i])
                    outputs.append(text_list)
                else:
                    outputs.append(examples[k])
        tokenized_examples = tokenizer(
            *inputs, truncation=True, padding="max_length", max_length=128
        )
        tokenized_labels = tokenizer(
            *outputs, truncation=True, padding="max_length", max_length=128
        )
        tokenized_examples["labels"] = tokenized_labels["input_ids"]
        return tokenized_examples

    tokenized_dataset = lineage_dataset.get_dataset().map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=lineage_dataset.get_dataset().column_names,
        load_from_cache_file=False,
    )
    return LineageDataset(
        dataset=tokenized_dataset, feature_keys=lineage_dataset.feature_keys
    )


def glue_preprocess_function(lineage_dataset, tokenizer, **kwargs):
    init_dataset = lineage_dataset.get_dataset()
    feature_keys = lineage_dataset.feature_keys

    def preprocess_function(examples):
        inputs = [examples[k] for k in feature_keys]
        return tokenizer(*inputs, truncation=True, max_length=128)

    encoded_dataset = init_dataset.map(
        preprocess_function,
        num_proc=4,
        batched=True,
        load_from_cache_file=False,
    )
    return LineageDataset(
        dataset=encoded_dataset, feature_keys=lineage_dataset.feature_keys
    )


# -

def compute_metrics_glue(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metric = load_metric('glue', 'sst2')
    return metric.compute(predictions=predictions, references=labels)

def compute_metrics_mlm(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    indices = [[i for i, x in enumerate(labels[row]) if x != -100] for row in range(len(labels))]

    labels = [labels[row][indices[row]] for row in range(len(labels))]
    labels = [item for sublist in labels for item in sublist]

    predictions = [predictions[row][indices[row]] for row in range(len(predictions))]
    predictions = [item for sublist in predictions for item in sublist]

    metric = load_metric('accuracy')
    return metric.compute(predictions=predictions, references=labels)


def vanilla_finetune_init_function(cur_node, parent_list):
    
    def register_init_from(src_node, dst_node, src_obj_name=None, dst_obj_name=None, name_map=None):
        if name_map:
            convert = lambda x: name_map[x]
        else:
            convert = lambda x: x

        if src_obj_name is None and dst_obj_name is None:

            def equal(x, y):
                return x[1] == y[1]

            src_layers = list(src_node.get_pt_model().state_dict().keys())
            dst_layers = list(dst_node.get_pt_model().state_dict().keys())

            a = [
                (name, src_node.get_pt_model().state_dict()[name].shape)
                for name in src_layers
            ]
            b = [
                (name, dst_node.get_pt_model().state_dict()[name].shape)
                for name in dst_layers
            ]

            name_map = {src_layers[p[0]]: dst_layers[p[1]] for p in lcs_one(a, b, equal)}
            for name in name_map.keys():
                register_init_from(
                    src_node,
                    dst_node,
                    src_obj_name=name,
                    dst_obj_name=name_map[name],
                    name_map=name_map,
                )
            return

        else:
            obj = src_node.get_model().get_state_reference(src_obj_name)
            copy_obj = copy.deepcopy(obj)
            dst_node.get_model().set_state_reference(dst_obj_name, copy_obj)
    
            src_local_names = src_node.get_local_ids_of_obj(src_obj_name)
            graph = src_node.graph
            for name in src_local_names:
                dst_node.local_to_global_id[convert(name)] = dst_node.create_global_id(
                    convert(name)
                )
                graph.init_tracker.set_init_from(
                    dst_node.local_to_global_id[convert(name)],
                    src_node.local_to_global_id[name],
                )
        
    
    register_init_from(parent_list[0],cur_node)

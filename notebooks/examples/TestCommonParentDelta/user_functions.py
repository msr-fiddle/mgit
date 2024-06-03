import numpy as np
from datasets import load_metric
import copy
import os
import sys
MGIT_PATH=os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(MGIT_PATH)
from utils.lineage.graph import *

def test_success_condition(eval_results):
    print(eval_results)
    return eval_results['eval_accuracy'] > 0.70

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metric = load_metric('glue', 'mnli')
    return metric.compute(predictions=predictions, references=labels)

def vanilla_finetune_init_function(cur_node,parent_list):
    #parent_model = parent_list[0].get_pt_model()
    #cur_modelInst.model = copy.deepcopy(parent_model)
    INIT_FROM(parent_list[0],cur_node)
    #parent_tokenizer = parent_list[0].get_pt_tokenizer()
    #cur_node.modelInst.tokenizer = copy.deepcopy(parent_tokenizer)
    
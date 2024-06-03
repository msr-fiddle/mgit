import numpy as np
from datasets import load_metric
import copy
import os
import sys
MGIT_PATH=os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(MGIT_PATH)
from utils.lineage.graph import *
from utils.lcs.diffcheck import lcs_one

def test_success_condition(eval_results):
    print(eval_results)
    return eval_results['eval_accuracy'] > 0.70

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metric = load_metric('glue', 'mnli')
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

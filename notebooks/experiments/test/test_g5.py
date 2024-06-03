import os
import sys
MGIT_PATH=os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(MGIT_PATH)
from utils.lineage.graph import *
from utils import meta_functions
import argparse
import time
import torch
import shutil

parser = argparse.ArgumentParser(description="G5 Test")
parser.add_argument(
    "-g", "--g-path", default="/datadrive3/mgit/g5", type=str, help="path to g5"
)
parser.add_argument(
    "--param_home", default="./parameter_store/", type=str, help="location to store hashed parameters"
)
parser.add_argument(
    "--compression_mode", default="lzma", type=str, help="storage compression mode"
)
parser.add_argument("--single_model_compression", default=False, action="store_true")
parser.add_argument("--is_delta", default=True, action="store_false")
parser.add_argument("--no_quantize_delta", default=True, action="store_false", dest="quantize_delta")


def get_storage_space(path_name):
    str_ret = os.popen("du -s " + path_name).read()
    store_amount = int(str_ret.split("\t")[0])
    return store_amount


def compute_storage_savings(graph):
    return sum(
        get_storage_space(node_name) for node_name in graph.nodes
    ) / get_storage_space("parameter_store")


datasets = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
perturbation_file = MGIT_PATH + "/utils/perturbations/perturbation_utils.py"
user_file = os.getcwd() + "/user_functions.py"

models = ["roberta-base"]
datasets = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]


task_to_keys = {
    "cola": ["sentence"],
    "mnli": ["premise", "hypothesis"],
    "mnli-mm": ["premise", "hypothesis"],
    "mrpc": ["sentence1", "sentence2"],
    "qnli": ["question", "sentence"],
    "qqp": ["question1", "question2"],
    "rte": ["sentence1", "sentence2"],
    "sst2": ["sentence"],
    "stsb": ["sentence1", "sentence2"],
    "wnli": ["sentence1", "sentence2"],
}

# perturbations=['','char_delete','word_order','char_lettercase','char_replace','addtypos','char_swap','char_misspelledword','char_insert','char_repetition']
perturbations = [""]


def main(args):
    shutil.rmtree("parameter_store", ignore_errors=True)
    model_home = args.g_path
    compression_mode = args.compression_mode
    single_model_compression = args.single_model_compression
    quantize_delta = args.quantize_delta
    param_home = args.param_home

    g = LineageGraph(
        compression_mode=compression_mode,
        single_model_compression=single_model_compression,
        param_home=param_home,
    )
    for perturbation in perturbations:
        for task in datasets:
            validation_key = (
                "validation_mismatched"
                if task == "mnli-mm"
                else "validation_matched"
                if task == "mnli"
                else "validation"
            )
            if perturbation == "":
                test_name = task
                l_dataset = LineageDataset(
                    "glue",
                    task,
                    split=validation_key,
                    cache_dir="./.cache",
                    feature_keys=task_to_keys[task],
                )
            else:
                test_name = task + "_" + perturbation
                l_dataset = LineageDataset(
                    "glue",
                    task,
                    preprocess_function_name="perturb_" + perturbation,
                    preprocess_function_path=perturbation_file,
                    split=validation_key,
                    cache_dir="./.cache",
                    feature_keys=task_to_keys[task],
                )
            metric_name = (
                "pearson"
                if task == "stsb"
                else "matthews_correlation"
                if task == "cola"
                else "accuracy"
            )
            test = LineageTest(
                custom_test_function_path=user_file,
                custom_test_function_name="glue_custom_test_function",
                test_success_condition_path=user_file,
                test_success_condition_name=task + "_test_success_condition",
                eval_dataset=l_dataset,
                metric_for_best_model=metric_name,
                name=test_name,
            )
            g.register_test_to_type(test, task)

    root = LineageNode(
        init_checkpoint="roberta-base",
        output_dir=model_home + "roberta-base",
        is_delta=args.is_delta,
        quantize_delta = quantize_delta,
    )
    g.add(root)

    for model in models:
        for dset in datasets:
            next_parent = model_home + "roberta-base"
            full_name = model_home + model + "_" + dset + "_MTL" + "-" + str(1)
            node = LineageNode(
                output_dir=full_name, model_type=dset, is_delta=args.is_delta, quantize_delta=quantize_delta,
            )
            g.add(node, etype="adapted", parent=next_parent)
            if next_parent != model_home + "roberta-base":
                g.add(node, etype="versioned", parent=next_parent)
            next_parent = full_name

    start_t = time.time()
    root.get_model().save(vanilla_save=False)
    # root.unload_model(save_model=False)
    count = 0
    for model in models:
        for dset in datasets:
            full_name = model_home + model + "_" + dset + "_MTL" + "-" + str(1)
            node = g.get_node(full_name)
            if count == 0:
                node.get_model().save(vanilla_save=False)
                new_roberta = node.get_pt_model().roberta
            else:
                pt_model = node.get_pt_model()
                pt_model.roberta = None
                torch.save(
                    pt_model.state_dict(),
                    os.path.join(g.param_home, "head-" + str(count)),
                )
                pt_model.roberta = new_roberta
            # node.unload_model(save_model=False)
            count += 1
    save_time = time.time() - start_t

    start_t = time.time()
    count = 0
    for model in models:
        for dset in datasets:
            full_name = model_home + model + "_" + dset + "_MTL" + "-" + str(1)
            node = g.get_node(full_name)
            if count > 0 or (not args.is_delta and not single_model_compression):
                node.run_all_tests()
            count += 1
    if args.is_delta or single_model_compression:
        save_time += time.time() - start_t

    for task in datasets:
        nodes = [
            node.output_dir for node in g.nodes.values() if node.model_type == task
        ]
        print(meta_functions.show_result_table(g, nodes, show_metrics=True))

    try:
        print("Storage time:", save_time)
        print(f"Storage savings: {compute_storage_savings(g):.3f}")
    except:
        pass


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

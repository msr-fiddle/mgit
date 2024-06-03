import os
import sys
MGIT_PATH=os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(MGIT_PATH)
from utils.lineage.graph import *
from utils import meta_functions
import argparse
import time
import shutil


def get_storage_space(path_name):
    str_ret = os.popen("du -s " + path_name).read()
    store_amount = int(str_ret.split("\t")[0])
    return store_amount


def compute_storage_savings(graph):
    return sum(
        get_storage_space(node_name) for node_name in graph.nodes
    ) / get_storage_space("parameter_store")


models = ["roberta-base"]
adaptations = ["vanilla-finetune"]
datasets = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
versions = [""] + ["v" + str(i) for i in range(1, 10)]

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

user_file = os.getcwd() + "/user_functions.py"

parser = argparse.ArgumentParser(description="G2 Construction")
parser.add_argument(
    "-g",
    "--g-path",
    default="/datadrive3/mgit/g2",
    type=str,
    help="path containing g2 models",
)
parser.add_argument("--skip_save_models", default=False, action="store_true")
parser.add_argument(
    "--compression_mode", default="lzma", type=str, help="storage compression mode"
)
parser.add_argument("--single_model_compression", default=False, action="store_true")
parser.add_argument("--is_delta", default=True, action="store_false")
parser.add_argument("--no_quantize_delta", default=True, action="store_false", dest="quantize_delta")


def main(args):
    shutil.rmtree("parameter_store", ignore_errors=True)
    start_time = time.time()
    model_home = args.g_path
    compression_mode = args.compression_mode
    single_model_compression = args.single_model_compression

    task_to_dataset = {}
    for task in datasets:
        validation_key = (
            "validation_mismatched"
            if task == "mnli-mm"
            else "validation_matched"
            if task == "mnli"
            else "validation"
        )
        l_dataset = LineageDataset(
            "glue",
            task,
            split=validation_key,
            cache_dir="./.cache",
            feature_keys=task_to_keys[task],
        )
        task_to_dataset[task] = l_dataset

    g = LineageGraph(
        compression_mode=compression_mode,
        single_model_compression=single_model_compression,
    )

    for task in datasets:
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
            eval_dataset=task_to_dataset[task],
            metric_for_best_model=metric_name,
            name=task,
        )
        g.register_test_to_type(test, task)

    root = LineageNode(
        init_checkpoint="roberta-base",
        output_dir="roberta-base",
        is_delta=args.is_delta, 
        quantize_delta = args.quantize_delta,
    )
    g.add(root)
    if not args.skip_save_models:
        root.get_model().save()

    for model in models:
        for adapt in adaptations:
            for dset in datasets:
                next_parent = "roberta-base"
                for vers in versions:
                    full_name = model_home + model + "_" + adapt + "_" + dset
                    if vers != "":
                        full_name += "_" + vers
                    node = LineageNode(
                        output_dir=full_name, model_type=dset, is_delta=args.is_delta, quantize_delta = args.quantize_delta,
                    )
                    g.add(node, etype="adapted", parent=next_parent)
                    if next_parent != "roberta-base":
                        g.add(node, etype="versioned", parent=next_parent)
                    if not args.skip_save_models:
                        node.get_model().save()
                    node.unload_model(save_model=False)
                    next_parent = full_name

    for task in datasets:
        nodes = [
            node.output_dir for node in g.nodes.values() if node.model_type == task
        ]
        print(meta_functions.show_result_table(g, nodes, show_metrics=True))

    try:
        print(f"Storage savings: {compute_storage_savings(g):.3f}")
    except:
        pass

    print(f"Total time: {(time.time() - start_time) / 3600.0:.3f} hours")

    g.show()
    g.save("./", save_models=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

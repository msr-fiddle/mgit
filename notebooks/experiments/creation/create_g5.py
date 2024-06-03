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
    print(str_ret)
    store_amount = int(str_ret.split("\t")[0])
    return store_amount


def compute_storage_savings(graph):
    return sum(
        get_storage_space(node_name) for node_name in graph.nodes
    ) / get_storage_space("parameter_store")


TRANSFORMERS_CACHE = '/workspace/HF_cache/transformers_cache/'

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

preprocess_file = MGIT_PATH + "/utils/preprocess_utils.py"
user_file = os.getcwd() + "/user_functions.py"
perturbation_file = MGIT_PATH + "/utils/perturbation_utils.py"

parser = argparse.ArgumentParser(description="G5 Construction")
#parser.add_argument(
#    "-g",
#    "--g-path",
#    default="./",
#    type=str,
#    help="path containing g5 models",
#)
parser.add_argument("--skip_save_models", default=False, action="store_true")
parser.add_argument(
    "--compression_mode", default="lzma", type=str, help="storage compression mode"
)
parser.add_argument("--single_model_compression", default=False, action="store_true")
parser.add_argument("--is_delta", default=True, action="store_false")
parser.add_argument("--run_training", default=False, action="store_true")
parser.add_argument("--no_quantize_delta", default=True, action="store_false", dest="quantize_delta")


def main(args):
    shutil.rmtree("parameter_store", ignore_errors=True)
    start_time = time.time()
    #model_home = args.g_path
    compression_mode = args.compression_mode
    single_model_compression = args.single_model_compression
    quantize_delta = args.quantize_delta
    is_delta = args.is_delta
    is_training = args.run_training

    task_to_dataset = {}
    train_task_to_dataset = {}
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
        train_dataset = LineageDataset(
            "glue",
            task,
            split="train",
            cache_dir="./.cache",
            feature_keys=task_to_keys[task],
        )
        task_to_dataset[task] = l_dataset
        train_task_to_dataset[task] = train_dataset

    g = LineageGraph(
        compression_mode=compression_mode,
        single_model_compression=single_model_compression,
    )

    root = LineageNode(
        init_checkpoint="roberta-base",
        output_dir="roberta-base",
        is_delta=is_delta,
        quantize_delta = quantize_delta,
    )

    g.add(root)

    for model_name in models:
        for dset in datasets:
            if is_training:
                lineage_train = LineageTrain(
                    preprocess_function_path=preprocess_file,
                    preprocess_function_name="glue_preprocess_function",
                    train_dataset=train_task_to_dataset[dset],
                    eval_dataset=task_to_dataset[dset],
                    learning_rate=2e-05,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    num_train_epochs=0.001,
                    # metric_for_best_model= matthews_correlation,
                    # compute_metrics_path= user_functions.py,
                    # compute_metrics_name= cola_compute_metrics
                )
            else:
                lineage_train = None
            full_name = model_name + "_" + dset + "_MTL"
            node = LineageNode(
                model_init_function_path=user_file,
                model_init_function_name="model_init_function",
                output_dir=full_name,
                init_checkpoint="roberta-base",
                lineage_train=lineage_train,
                model_type=dset,
                is_delta=is_delta,
                quantize_delta = quantize_delta,
            )
            g.add(node, etype="adapted", parent="roberta-base")
            node.get_model()
            node.unload_model(save_model=False)

    share_set = [
        "roberta-base_cola_MTL",
        "roberta-base_mnli_MTL",
        "roberta-base_mrpc_MTL",
        "roberta-base_qnli_MTL",
        "roberta-base_qqp_MTL",
        "roberta-base_rte_MTL",
        "roberta-base_sst2_MTL",
        "roberta-base_stsb_MTL",
        "roberta-base_wnli_MTL",
    ]

    assert set(share_set) == set(
        g.entanglement_tracker.entangle_map[
            "roberta-base_cola_MTL-roberta.embeddings.position_ids"
        ]
    )

    for node in g.nodes.values():
        node.train()
        if not args.skip_save_models:
            node.get_model().save()
    time.sleep(5)
    print(meta_functions.show_result_table(g, show_metrics=True))

    print(f"Storage savings: {compute_storage_savings(g):.3f}")
    print(f"Total time: {(time.time() - start_time) / 3600.0:.3f} hours")

    g.save("./", save_models=False)
    n = g.show()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

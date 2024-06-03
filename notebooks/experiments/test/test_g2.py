import os
import sys
MGIT_PATH=os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(MGIT_PATH)
from utils.lineage.graph import *
from utils import meta_functions
import argparse
import time

parser = argparse.ArgumentParser(description="G2 Test")
parser.add_argument(
    "-g", "--g-path", default="/datadrive3/mgit/g2", type=str, help="path to g2"
)

datasets = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
perturbation_file = MGIT_PATH + "/utils/perturbations/perturbation_utils.py"
user_file = os.getcwd() + "/user_functions.py"

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

perturbations = [
    "",
    "char_delete",
    "word_order",
    "char_lettercase",
    "char_replace",
    "addtypos",
    "char_swap",
    "char_misspelledword",
    "char_insert",
    "char_repetition",
]


def main(args):
    model_home = args.g_path
    g = LineageGraph()
    print(versions)
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
    )
    g.add(root)
    root.get_model().save()

    for model in models:
        for adapt in adaptations:
            for dset in datasets:
                next_parent = model_home + "roberta-base"
                for vers in versions:
                    full_name = model_home + model + "_" + adapt + "_" + dset
                    if vers != "":
                        full_name += "_" + vers
                    node = LineageNode(output_dir=full_name, model_type=dset)
                    g.add(node, etype="adapted", parent=next_parent)
                    if next_parent != model_home + "roberta-base":
                        g.add(node, etype="versioned", parent=next_parent)
                    # node.run_all_tests()
                    node.unload_model(save_model=False)
                    next_parent = full_name

    bisect_runtimes = []
    for task in datasets:
        start_t = time.time()
        leaf_name = model_home + "roberta-base_vanilla-finetune_" + task + "_v9"
        start_name = model_home + "roberta-base_vanilla-finetune_" + task
        print(
            g.get_node(leaf_name).test_name_list, g.get_node(start_name).test_name_list
        )
        meta_functions.bisect(task, g.get_node(leaf_name), g.get_node(start_name))
        bisect_runtimes.append(time.time() - start_t)

    baseline_runtimes = []
    for task in datasets:
        start_t = time.time()
        leaf_name = model_home + "roberta-base_vanilla-finetune_" + task + "_v9"
        cur_node = g.get_node(leaf_name)
        cur_model_type = cur_node.model_type
        test_success = cur_node.run_test_by_name(task)
        assert isinstance(test_success, bool)
        cur_node.unload_model(save_model=False)
        while not test_success:
            cur_node = cur_node.get_parents("adapted")[0]
            if cur_node.model_type != cur_model_type:
                break
            test_success = cur_node.run_test_by_name(task)
            cur_node.unload_model(save_model=False)
            assert isinstance(test_success, bool)
        baseline_runtimes.append(time.time() - start_t)

    for task in datasets:
        nodes = [
            node.output_dir for node in g.nodes.values() if node.model_type == task
        ]
        print(meta_functions.show_result_table(g, nodes, show_metrics=True))

    for i in range(len(datasets)):
        print(
            datasets[i],
            "baseline runtime:",
            baseline_runtimes[i],
            "bisect runtime:",
            bisect_runtimes[i],
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

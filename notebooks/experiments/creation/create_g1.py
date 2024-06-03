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


TRANSFORMERS_CACHE = '/workspace/HF_cache/transformers_cache/'

preprocess_file = MGIT_PATH + "/utils/preprocess_utils.py"
user_file = os.getcwd() + "/user_functions.py"
perturbation_file = MGIT_PATH + "/utils/perturbation_utils.py"

parser = argparse.ArgumentParser(description="G1 Construction")
parser.add_argument(
    "-g",
    "--g-path",
    default="/datadrive3/mgit/g1",
    type=str,
    help="path containing g1 models",
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

    bert_mlm_cased = model_home + "bert-base-cased"  # bert-base-cased Nov 15, 2018
    bert_mlm_uncased = (
        model_home + "bert-base-uncased"
    )  # bert-base-uncased Nov 15, 2018

    bert_squad2_uncased_frozen = (
        model_home + "ericRosello/bert-base-uncased-finetuned-squad-frozen-v2"
    )  # ericRosello/bert-base-uncased-finetuned-squad-frozen-v2
    # Jan 4, 2022
    bert_mnli = (
        model_home + "aloxatel/bert-base-mnli"
    )  # aloxatel/bert-base-mnli Jul 28, 2020
    bert_mlm_large_cased = (
        model_home + "bert-large-cased"
    )  # bert-large-cased Nov 30, 2018
    bert_mlm_large_uncased = (
        model_home + "bert-large-uncased"
    )  # bert-large-uncased Nov 15, 2018
    bert_mnli_large = (
        model_home + "TehranNLP-org/bert-large-mnli"
    )  # TehranNLP-org/bert-large-mnli Apr 30, 2022
    bert_squad2_uncased = (
        model_home + "deepset/bert-base-uncased-squad2"
    )  # deepset/bert-base-uncased-squad2 Jan 14, 2022

    roberta_mlm_large = model_home + "roberta-large"  # roberta-large  Aug 5, 2019
    roberta_mnli_large = (
        model_home + "roberta-large-mnli"
    )  # roberta-large-mnli Aug 5, 2019
    roberta_squad2_large = (
        model_home + "deepset/roberta-large-squad2"
    )  # deepset/roberta-base-squad2 Jan 22, 2020
    roberta_mlm = model_home + "roberta-base"  # roberta-base Aug 4, 2019
    roberta_mnli = (
        model_home + "textattack/roberta-base-MNLI"
    )  # textattack/roberta-base-MNLI Jun 7, 2020
    roberta_squad2 = (
        model_home + "deepset/roberta-base-squad2"
    )  # deepset/roberta-base-squad2 Jan 22, 2020

    albert_mlm = model_home + "albert-base-v2"  # albert-base-v2 Nov 4, 2019
    albert_mnli = (
        model_home + "prajjwal1/albert-base-v2-mnli"
    )  # prajjwal1/albert-base-v2-mnli May 26, 2020
    albert_squad2 = (
        model_home + "twmkn9/albert-base-v2-squad2"
    )  # twmkn9/albert-base-v2-squad2 Mar 9, 2020

    distilbert_mlm_cased = (
        model_home + "distilbert-base-cased"
    )  # distilbert-base-cased Feb 7, 2020
    distilbert_mlm_uncased = (
        model_home + "distilbert-base-uncased"
    )  # distilbert-base-uncased Aug 28, 2019
    distilbert_squad2_uncased = (
        model_home + "twmkn9/distilbert-base-uncased-squad2"
    )  # twmkn9/distilbert-base-uncased-squad2 Mar 23, 2020
    distilbert_squad2_uncased_frozen = (
        model_home + "ericRosello/distilbert-base-uncased-finetuned-squad-frozen-v2"
    )  # ericRosello/distilbert-base-uncased-finetuned-squad-frozen-v2
    # Jan 4, 2022

    electra_mlm_small = (
        model_home + "google/electra-small-generator"
    )  # google/electra-small-generator Mar 24, 2020
    # electra_mlm_base = './models/electra/mlm_base' #google/electra-base-generator Mar 24, 2020
    # electra_mlm_large = './models/electra/mlm_large' #google/electra-large-generator Mar 24, 2020
    electra_mnli_small = (
        model_home + "howey/electra-small-mnli"
    )  # howey/electra-small-mnli Apr 15, 2021
    # electra_mnli_base = './models/electra/mnli_base'#howey/electra-base-mnli
    # electra_mnli_large = './models/electra/mnli_large' #howey/electra-large-mnli
    # electra_squad2_base = './models/electra/squad2_base'#deepset/electra-base-squad2

    model_types = dict(
        [
            (bert_mlm_cased, "mlm"),
            (bert_mlm_uncased, "mlm"),
            (bert_squad2_uncased_frozen, "squad"),
            (bert_mnli, "mnli"),
            (bert_mlm_large_cased, "mlm"),
            (bert_mlm_large_uncased, "mlm"),
            (bert_mnli_large, "mnli"),
            (bert_squad2_uncased, "squad"),
            (roberta_mlm_large, "mlm"),
            (roberta_mnli_large, "mnli"),
            (roberta_squad2_large, "squad"),
            (roberta_mlm, "mlm"),
            (roberta_mnli, "mnli"),
            (roberta_squad2, "squad"),
            (albert_mlm, "mlm"),
            (albert_mnli, "mnli"),
            (albert_squad2, "squad"),
            (distilbert_mlm_cased, "mlm"),
            (distilbert_mlm_uncased, "mlm"),
            (distilbert_squad2_uncased, "squad"),
            (distilbert_squad2_uncased_frozen, "squad"),
            (electra_mlm_small, "mlm"),
            (electra_mnli_small, "mnli"),
        ]
    )

    task_types = dict(
        [
            (bert_mlm_cased, "MaskedLM"),
            (bert_mlm_uncased, "MaskedLM"),
            (bert_squad2_uncased_frozen, "question_answering"),
            (bert_mnli, "sequence_classification"),
            (bert_mlm_large_cased, "MaskedLM"),
            (bert_mlm_large_uncased, "MaskedLM"),
            (bert_mnli_large, "sequence_classification"),
            (bert_squad2_uncased, "question_answering"),
            (roberta_mlm_large, "MaskedLM"),
            (roberta_mnli_large, "sequence_classification"),
            (roberta_squad2_large, "question_answering"),
            (roberta_mlm, "MaskedLM"),
            (roberta_mnli, "sequence_classification"),
            (roberta_squad2, "question_answering"),
            (albert_mlm, "MaskedLM"),
            (albert_mnli, "sequence_classification"),
            (albert_squad2, "question_answering"),
            (distilbert_mlm_cased, "MaskedLM"),
            (distilbert_mlm_uncased, "MaskedLM"),
            (distilbert_squad2_uncased, "question_answering"),
            (distilbert_squad2_uncased_frozen, "question_answering"),
            (electra_mlm_small, "MaskedLM"),
            (electra_mnli_small, "sequence_classification"),
        ]
    )

    model_pool = [
        bert_mlm_cased,
        bert_mlm_uncased,
        bert_mnli,
        bert_squad2_uncased_frozen,
        bert_squad2_uncased,
        bert_mlm_large_uncased,
        bert_mlm_large_cased,
        bert_mnli_large,
        roberta_mlm,
        roberta_squad2,
        roberta_mnli,
        roberta_mlm_large,
        roberta_mnli_large,
        roberta_squad2_large,
        albert_mlm,
        albert_squad2,
        albert_mnli,
        distilbert_mlm_uncased,
        distilbert_mlm_cased,
        distilbert_squad2_uncased,
        distilbert_squad2_uncased_frozen,
        electra_mlm_small,
        electra_mnli_small,
    ]

    mlm_lineage_eval_dataset = LineageDataset(
        "wikitext",
        "wikitext-103-raw-v1",
        cache_dir=TRANSFORMERS_CACHE,
        split="validation",
        feature_keys=["text"],
    )

    squad_lineage_eval_dataset = LineageDataset(
        "squad_v2",
        cache_dir=TRANSFORMERS_CACHE,
        split="validation",
        feature_keys=["context", "question"],
    )

    squadv1_lineage_eval_dataset = LineageDataset(
        "squad",
        cache_dir=TRANSFORMERS_CACHE,
        split="validation",
        feature_keys=["context", "question"],
    )

    mnli_lineage_eval_dataset = LineageDataset(
        "glue",
        "mnli",
        split="validation_matched",
        cache_dir=TRANSFORMERS_CACHE,
        feature_keys=["premise", "hypothesis"],
    )

    mlm_test = LineageTest(
        preprocess_function_path=preprocess_file,
        preprocess_function_name="mlm_preprocess_function",
        eval_dataset=mlm_lineage_eval_dataset,
        metric_for_best_model="loss",
        name="mlm",
    )

    mnli_test = LineageTest(
        custom_test_function_path=user_file,
        custom_test_function_name="mnli_custom_test_function",
        eval_dataset=mnli_lineage_eval_dataset,
        metric_for_best_model="accuracy",
        name="mnli",
    )

    squadv2_test = LineageTest(
        preprocess_function_path=preprocess_file,
        preprocess_function_name="squad_preprocess_validation_function",
        eval_dataset=squad_lineage_eval_dataset,
        postprocess_function_path=preprocess_file,
        postprocess_function_name="postprocess_squad2_predictions",
        metric_for_best_model="f1",
        name="squad_v2",
    )

    squadv1_test = LineageTest(
        preprocess_function_path=preprocess_file,
        preprocess_function_name="squad_preprocess_validation_function",
        eval_dataset=squadv1_lineage_eval_dataset,
        postprocess_function_path=preprocess_file,
        postprocess_function_name="postprocess_squad_predictions",
        metric_for_best_model="f1",
        name="squad_v1",
    )

    g = LineageGraph(
        compression_mode=compression_mode,
        single_model_compression=single_model_compression,
    )
    g.register_test_to_type(mlm_test, "mlm")
    g.register_test_to_type(mnli_test, "mnli")
    g.register_test_to_type(squadv2_test, "squad")
    g.register_test_to_type(squadv1_test, "squad")

    for model in model_pool:
        print("Inserting: " + model)
        node = LineageNode(
            output_dir=model,
            init_checkpoint=model,
            model_type=model_types[model],
            task_type=task_types[model],
            is_delta=args.is_delta,
            quantize_delta = args.quantize_delta,
        )
        if not g.add(node):
            g.add_root(node)
        if not args.is_delta:
            g.get_node(model).run_all_tests()
        if not args.skip_save_models:
            node.get_model().save()
        for ex_node in g.nodes.values():
            ex_node.unload_model(save_model=False)

    print(meta_functions.show_result_table(g, show_metrics=True))

    print(f"Storage savings: {compute_storage_savings(g):.3f}")
    print(f"Total time: {(time.time() - start_time) / 3600.0:.3f} hours")

    g.save("./", save_models=False)
    n = g.show()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

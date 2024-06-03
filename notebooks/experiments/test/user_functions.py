import numpy as np
from transformers import Trainer, TrainingArguments
from datasets import load_metric


def mnli_custom_test_function(model, lineage_dataset, tokenizer):
    dataset = lineage_dataset.get_dataset()
    feature_keys = lineage_dataset.feature_keys

    custom_label_map = {0: 0, 1: 1, 2: 2}
    dataset_label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    if hasattr(model.config, "id2label"):
        id2label = model.config.id2label
        if set(dataset_label2id.keys()) == set(
            [label_name.lower() for label_name in id2label.values()]
        ):
            for k, v in id2label.items():
                custom_label_map[dataset_label2id[v.lower()]] = int(k)

    def label_swapping(example):
        example["label"] = custom_label_map[example["label"]]
        return example

    swap_dataset = dataset.map(label_swapping)

    def preprocess_function(examples):
        inputs = [examples[k] for k in feature_keys]
        return tokenizer(*inputs, truncation=True, max_length=128)

    tokenized_dataset = swap_dataset.map(
        preprocess_function,
        num_proc=4,
        batched=True,
    )

    def mnli_compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return load_metric("glue", "mnli").compute(
            predictions=predictions, references=labels
        )

    trainer = Trainer(
        model,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=mnli_compute_metrics,
    )
    return trainer.evaluate()


def glue_custom_test_function(model, lineage_dataset, tokenizer):
    dataset = lineage_dataset.get_dataset()
    feature_keys = lineage_dataset.feature_keys
    task = model.config.model_type

    if task == "mnli":  # fix mismatching labels between dataset and model
        custom_label_map = {0: 0, 1: 1, 2: 2}
        dataset_label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
        if hasattr(model.config, "id2label"):
            id2label = model.config.id2label
            if set(dataset_label2id.keys()) == set(
                [label_name.lower() for label_name in id2label.values()]
            ):
                for k, v in id2label.items():
                    custom_label_map[dataset_label2id[v.lower()]] = int(k)

        def label_swapping(example):
            example["label"] = custom_label_map[example["label"]]
            return example

        proc_dataset = dataset.map(label_swapping)
    else:
        proc_dataset = dataset

    def preprocess_function(examples):
        inputs = [examples[k] for k in feature_keys]
        return tokenizer(*inputs, truncation=True, max_length=128)

    tokenized_dataset = proc_dataset.map(
        preprocess_function,
        num_proc=4,
        batched=True,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return load_metric("glue", task).compute(
            predictions=predictions, references=labels
        )

    args = TrainingArguments(
        "tmp",
        per_device_eval_batch_size=128,
    )

    trainer = Trainer(
        model,
        args,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer.evaluate()


def cola_test_success_condition(eval_results):
    print(eval_results)
    return eval_results["eval_matthews_correlation"] > 0.57


def mnli_test_success_condition(eval_results):
    print(eval_results)
    return eval_results["eval_accuracy"] > 0.86


def mrpc_test_success_condition(eval_results):
    print(eval_results)
    return eval_results["eval_accuracy"] > 0.89


def qnli_test_success_condition(eval_results):
    print(eval_results)
    return eval_results["eval_accuracy"] > 0.9


def qqp_test_success_condition(eval_results):
    print(eval_results)
    return eval_results["eval_accuracy"] > 0.9


def rte_test_success_condition(eval_results):
    print(eval_results)
    return eval_results["eval_accuracy"] > 0.78


def sst2_test_success_condition(eval_results):
    print(eval_results)
    return eval_results["eval_accuracy"] > 0.93


def stsb_test_success_condition(eval_results):
    print(eval_results)
    return eval_results["eval_pearson"] > 0.9


def wnli_test_success_condition(eval_results):
    print(eval_results)
    return eval_results["eval_accuracy"] > 0.54

import os
import torch
import transformers
import numpy as np
import sys
import json

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from adapter import *
from perturbations import perturbation_utils
from datasets import concatenate_datasets, load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

__all__ = [
    "TRANSFORMERS_CACHE",
    "load_tokenizers",
    "load_models",
    "task_to_keys",
]
TRANSFORMERS_CACHE = '/workspace/HF_cache/transformers_cache/'"
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def smart_embedding_resize(
    num_new_tokens,
    tokenizer_len,
    model,
):
    """Resize embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    if num_new_tokens != 0:
        model.resize_token_embeddings(tokenizer_len)
        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

def create_sc_model(mdckpt, num_labels, adapter=True, adapter_config=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        mdckpt, num_labels=num_labels, cache_dir=TRANSFORMERS_CACHE, low_cpu_mem_usage=True
    )
    if adapter and not hasattr(model.config, "adapter_size"):
        if adapter_config is None:
            adapter_config = {
                "adapter_size": 192,
                "adapter_act": "gelu",
                "adapter_initializer_range": 0.02,
            }
        add_config(model, adapter_config)
        add_adapters(model, model.config)

    return model


def load_tokenizers(checkpoint_filepaths):
    loaded_tokenizers = []
    for checkpoint_filepath in checkpoint_filepaths:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_filepath, use_fast=True, cache_dir=TRANSFORMERS_CACHE
        )
        loaded_tokenizers.append(tokenizer)
    return loaded_tokenizers


def load_models(checkpoint_filepaths, coarse=False, device = torch.device("cpu")):
    loaded_models = []
    tracing_module_pool = set()
    for checkpoint_filepath in checkpoint_filepaths:
        full_checkpoint_filepath = os.path.join(
            checkpoint_filepath, "pytorch_model.bin"
        )
        config = AutoConfig.from_pretrained(checkpoint_filepath)
        architecture = config.architectures[
            0
        ]  # TODO: Verify that this is not length > 1.

        if hasattr(config, "adapter_size"):
            model = getattr(transformers, architecture)(config)
            add_adapters(model, config)
            model.load_state_dict(
                torch.load(full_checkpoint_filepath, map_location=torch.device(device))
            )
        else:
            model = getattr(transformers, architecture).from_pretrained(
                checkpoint_filepath, low_cpu_mem_usage=True
            )

        if coarse:
            with open(
                os.path.join(checkpoint_filepath, "traced_modules"), "r"
            ) as traced_modules:
                tracing_module_pool.update(traced_modules.read().splitlines())

        loaded_models.append(model)

    return loaded_models, tracing_module_pool


def augment_dataset(dataset, feature_names, perturbations):
    perturbed_datasets = []
    for perturbation in perturbations:
        assert perturbation is not None
        if perturbation == "word_order":
            perturbed_dataset = perturbation_utils.perturb_word_order(
                dataset, feature_names
            )
        elif perturbation == "char_lettercase":
            perturbed_dataset = perturbation_utils.perturb_char_lettercase(
                dataset, feature_names
            )
        elif perturbation == "char_delete":
            perturbed_dataset = (
                perturbed_dataset
            ) = perturbation_utils.perturb_char_delete(dataset, feature_names)
        elif perturbation == "char_replace":
            perturbed_dataset = perturbation_utils.perturb_char_replace(
                dataset, feature_names
            )
        elif perturbation == "addtypos":
            perturbed_dataset = perturbation_utils.perturb_addtypos(
                dataset, feature_names
            )
        elif perturbation == "char_swap":
            perturbed_dataset = perturbation_utils.perturb_char_swap(
                dataset, feature_names
            )
        elif perturbation == "char_misspelledword":
            perturbed_dataset = perturbation_utils.perturb_char_misspelledword(
                dataset, feature_names
            )
        elif perturbation == "char_insert":
            perturbed_dataset = perturbation_utils.perturb_char_insert(
                dataset, feature_names
            )
        elif perturbation == "char_repetition":
            perturbed_dataset = perturbation_utils.perturb_char_repetition(
                dataset, feature_names
            )
        else:
            print(
                perturbation,
                "is not an implemented perturbation. Choose from: [word_order, char_lettercase, char_delete, char_replace, addtypos, char_swap, char_misspelledword, char_insert, char_repetition]",
            )
            assert False
        perturbed_datasets.append(perturbed_dataset)

    print(dataset)
    for split in dataset:
        for perturbed_dataset in perturbed_datasets:
            dataset[split] = concatenate_datasets(
                [dataset[split], perturbed_dataset[split]]
            )
    print(dataset)
    return dataset


def train_MLM(
    mdckpt,
    epochs=3,
    ds="wikitext",
    task="wikitext-103-raw-v1",
    batch_size=16,
    adapter=True,
    perturbations=[],
    sample_frac=1.0,
):
    assert (
        ds == "wikitext" and task == "wikitext-103-raw-v1"
    ), "currently only support wikitext dataset for mlm training"
    feature_names = ["text"]

    dataset = load_dataset(ds, task, cache_dir=TRANSFORMERS_CACHE)
    num_samples = int(len(dataset["train"]) * sample_frac)
    np.random.seed(0)
    random_select_idx = np.random.choice(
        len(dataset["train"]), num_samples, replace=False
    )
    assert len(set(random_select_idx.tolist())) == len(random_select_idx)
    print(
        "train set size:", len(dataset["train"]), "sample size:", len(random_select_idx)
    )
    dataset["train"] = dataset["train"].select(random_select_idx)
    for split in dataset:
        dataset[split] = dataset[split].add_column(
            "labels", dataset[split][feature_names[0]]
        )
    dataset = augment_dataset(dataset, feature_names, perturbations)

    model = transformers.AutoModelForMaskedLM.from_pretrained(
        mdckpt,
        cache_dir=TRANSFORMERS_CACHE,
        low_cpu_mem_usage=True
    )

    tokenizers = load_tokenizers([mdckpt])
    tokenizer = tokenizers[0]

    def tokenize_function(examples):
        inputs = [examples[k] for k in feature_names]
        if "labels" in examples:
            outputs = [examples["labels"]]
        else:
            outputs = []
            for k in feature_names:
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

    encoded_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )

    model_name = "tmp"

    STEPS_PER_EVAL = np.ceil(0.2 * len(encoded_dataset["train"]) / batch_size)

    args = TrainingArguments(
        model_name,
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=STEPS_PER_EVAL,
        eval_steps=STEPS_PER_EVAL,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        push_to_hub=False,
        seed=0,
    )
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    print(trainer.evaluate())

    return trainer


def train_sequence_classification(
    mdckpt,
    epochs=3,
    ds="glue",
    task="mnli",
    batch_size=16,
    adapter=True,
    perturbations=[],
):
    actual_task = "mnli" if task == "mnli-mm" else task
    sentence1_key, sentence2_key = task_to_keys[task]
    if sentence2_key is None:
        feature_names = [sentence1_key]
    else:
        feature_names = [sentence1_key, sentence2_key]

    dataset = load_dataset(ds, actual_task, cache_dir=TRANSFORMERS_CACHE)
    dataset = augment_dataset(dataset, feature_names, perturbations)

    metric = load_metric(ds, actual_task, cache_dir=TRANSFORMERS_CACHE)
    tokenizers = load_tokenizers([mdckpt])
    tokenizer = tokenizers[0]
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    model = create_sc_model(mdckpt, num_labels, adapter)
    if hasattr(model.config, "adapter_size"):
        freeze_all_parameters(model)
        unfreeze_adapters(model)

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(
            examples[sentence1_key], examples[sentence2_key], truncation=True
        )

    encoded_dataset = dataset.map(preprocess_function, num_proc=4, batched=True)

    metric_name = (
        "pearson"
        if task == "stsb"
        else "matthews_correlation"
        if task == "cola"
        else "accuracy"
    )
    model_name = "tmp"

    STEPS_PER_EVAL = np.ceil(0.2 * len(encoded_dataset["train"]) / batch_size)

    args = TrainingArguments(
        model_name,
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=STEPS_PER_EVAL,
        eval_steps=STEPS_PER_EVAL,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        push_to_hub=False,
        seed=0,
        metric_for_best_model=metric_name,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    validation_key = (
        "validation_mismatched"
        if task == "mnli-mm"
        else "validation_matched"
        if task == "mnli"
        else "validation"
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print(trainer.evaluate())

    return trainer


def train_and_save(
    save_path,
    init_mdckpt,
    epochs=3,
    ds="glue",
    task="mnli",
    batch_size=16,
    adapter=True,
    perturbations=[],
    sample_frac=1.0,
):
    if ds != "glue":
        trainer = train_MLM(
            init_mdckpt,
            epochs=epochs,
            ds=ds,
            task=task,
            batch_size=batch_size,
            adapter=adapter,
            perturbations=perturbations,
            sample_frac=sample_frac,
        )
    else:
        trainer = train_sequence_classification(
            init_mdckpt,
            epochs=epochs,
            ds=ds,
            task=task,
            batch_size=batch_size,
            adapter=adapter,
            perturbations=perturbations,
        )
    trainer.save_model(save_path)
    f = open(os.path.join(save_path, "train_log.json"), "w")
    json.dump(trainer.state.log_history, f)
    f.close()

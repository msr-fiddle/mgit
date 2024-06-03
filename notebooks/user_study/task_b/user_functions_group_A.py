import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['HF_HOME'] = '/workspace/HF_cache/'
os.environ['HF_DATASETS_CACHE'] = '/workspace/HF_cache/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/HF_cache/transformers_cache/'

import torch
import transformers
import datasets
import numpy as np
from datasets import load_metric

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


# -

#evaluation functions
def evaluate(model, tokenizer, dataset, compute_metrics):
    
    args = transformers.TrainingArguments(
                output_dir=os.path.join("tmp_trainer_args"),
            )
    if 'mlm' in str(compute_metrics):
        trainer = transformers.Trainer(
            model,
            args,
            tokenizer=tokenizer,
            eval_dataset=dataset,
            compute_metrics=compute_metrics,
            data_collator=transformers.DataCollatorForLanguageModeling(
                        tokenizer=tokenizer, mlm_probability=0.15
                    ),
        )
    else:
        trainer = transformers.Trainer(
            model,
            args,
            tokenizer=tokenizer,
            eval_dataset=dataset,
            compute_metrics=compute_metrics,
        )
    #trainer uses gpu to evaluate model
    eval_results = "%.3f" % trainer.evaluate()['eval_accuracy']
    return {'eval_accuracy': eval_results}


# +
#preprocess and postprocess functions for datasets
def mlm_preprocess_function(dataset, tokenizer):
    feature_keys = ['text']

    def tokenize_function(examples):
        inputs = [examples[k] for k in feature_keys]
        if "labels" in examples:
            outputs = [examples["labels"]]
        else:
            outputs = []
            for k in feature_keys:
                outputs.append(examples[k])
        tokenized_examples = tokenizer(
            *inputs, truncation=True, padding="max_length", max_length=128
        )
        tokenized_labels = tokenizer(
            *outputs, truncation=True, padding="max_length", max_length=128
        )
        tokenized_examples["labels"] = tokenized_labels["input_ids"]
        return tokenized_examples

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    return tokenized_dataset

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

def glue_preprocess_function(dataset, tokenizer):
    feature_keys = ['sentence']

    def preprocess_function(examples):
        inputs = [examples[k] for k in feature_keys]
        return tokenizer(*inputs, truncation=True, max_length=128)

    encoded_dataset = dataset.map(
        preprocess_function,
        num_proc=4,
        batched=True,
        load_from_cache_file=False,
    )
    return encoded_dataset

def compute_metrics_glue(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metric = load_metric('glue', 'sst2')
    return metric.compute(predictions=predictions, references=labels)

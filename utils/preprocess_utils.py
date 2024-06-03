import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from lineage.graph import *
from tqdm.auto import tqdm
import collections
import numpy as np
from datasets import load_metric

__all__ = [
    "mlm_preprocess_function",
    "glue_preprocess_function",
    "squad_preprocess_train_function",
    "squad_preprocess_validation_function",
    "postprocess_qa_predictions",
    "mnli_compute_metrics",
]


def mlm_preprocess_function(lineage_dataset, tokenizer, **kwargs):
    feature_keys = lineage_dataset.feature_keys

    def tokenize_function(examples):
        inputs = [examples[k] for k in feature_keys]
        if "labels" in examples:
            outputs = [examples["labels"]]
        else:
            outputs = []
            for k in feature_keys:
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

    tokenized_dataset = lineage_dataset.get_dataset().map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=lineage_dataset.get_dataset().column_names,
        load_from_cache_file=False,
    )
    return LineageDataset(
        dataset=tokenized_dataset, feature_keys=lineage_dataset.feature_keys
    )


def glue_preprocess_function(lineage_dataset, tokenizer, **kwargs):
    init_dataset = lineage_dataset.get_dataset()
    feature_keys = lineage_dataset.feature_keys

    def preprocess_function(examples):
        inputs = [examples[k] for k in feature_keys]
        return tokenizer(*inputs, truncation=True, max_length=128)

    encoded_dataset = init_dataset.map(
        preprocess_function,
        num_proc=4,
        batched=True,
        load_from_cache_file=False,
    )
    return LineageDataset(
        dataset=encoded_dataset, feature_keys=lineage_dataset.feature_keys
    )


def squad_preprocess_train_function(lineage_dataset, tokenizer, **kwargs):
    init_dataset = lineage_dataset.get_dataset()
    max_length = 384
    doc_stride = 128
    pad_on_right = tokenizer.padding_side == "right"

    def prepare_train_features(examples):
        examples["question"] = [q.lstrip() for q in examples["question"]]

        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    eval_dataset = init_dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=init_dataset.column_names,
        load_from_cache_file=False,
    )
    return LineageDataset(
        dataset=eval_dataset, feature_keys=lineage_dataset.feature_keys
    )


def squad_preprocess_validation_function(lineage_dataset, tokenizer, **kwargs):
    init_dataset = lineage_dataset.get_dataset()
    max_length = 384
    doc_stride = 128
    pad_on_right = tokenizer.padding_side == "right"

    def prepare_validation_features(examples):
        examples["question"] = [q.lstrip() for q in examples["question"]]
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = init_dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=init_dataset.column_names,
        load_from_cache_file=False,
    )
    eval_dataset.set_format(
        type=eval_dataset.format["type"],
        columns=list(eval_dataset.features.keys()),
    )
    return LineageDataset(
        dataset=eval_dataset, feature_keys=lineage_dataset.feature_keys
    )


def format_reference(dataset):
    formatted_references = []
    for ex in dataset:
        cur_dict = {}
        cur_dict["id"] = ex["id"]
        cur_dict["answers"] = {}
        cur_dict["answers"]["text"] = []
        for i in range(len(ex["answers"]["answer_start"])):
            cur_strt_idx = ex["answers"]["answer_start"][i]
            cur_end_idx = cur_strt_idx + len(ex["answers"]["text"][i])
            cur_dict["answers"]["text"].append(ex["context"][cur_strt_idx:cur_end_idx])
        cur_dict["answers"]["answer_start"] = ex["answers"]["answer_start"]
        formatted_references.append(cur_dict)
    return formatted_references


def postprocess_squad2_predictions(
    lineage_dataset, processed_lineage_dataset, raw_predictions, tokenizer, **kwargs
):
    n_best_size = 20
    max_answer_length = 30
    squad_v2 = True
    examples = lineage_dataset.get_dataset()
    features = processed_lineage_dataset.get_dataset()
    raw_predictions = raw_predictions.predictions
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    predictions = collections.OrderedDict()
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(
                tokenizer.cls_token_id
            )
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            best_answer = {"text": "", "score": 0.0}

        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = (
                best_answer["text"] if best_answer["score"] > min_null_score else ""
            )
            predictions[example["id"]] = answer

    final_predictions = predictions
    metric = load_metric("squad_v2" if squad_v2 else "squad")
    if squad_v2:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
            for k, v in final_predictions.items()
        ]
    else:
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in final_predictions.items()
        ]
    # references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    references = format_reference(examples)
    eval_results = metric.compute(
        predictions=formatted_predictions, references=references
    )
    return eval_results


def postprocess_squad_predictions(
    lineage_dataset, processed_lineage_dataset, raw_predictions, tokenizer, **kwargs
):
    n_best_size = 20
    max_answer_length = 30
    squad_v2 = False
    examples = lineage_dataset.get_dataset()
    features = processed_lineage_dataset.get_dataset()
    raw_predictions = raw_predictions.predictions
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    predictions = collections.OrderedDict()
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(
                tokenizer.cls_token_id
            )
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            best_answer = {"text": "", "score": 0.0}

        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = (
                best_answer["text"] if best_answer["score"] > min_null_score else ""
            )
            predictions[example["id"]] = answer

    final_predictions = predictions
    metric = load_metric("squad_v2" if squad_v2 else "squad")
    if squad_v2:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
            for k, v in final_predictions.items()
        ]
    else:
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in final_predictions.items()
        ]
    # references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    references = format_reference(examples)
    eval_results = metric.compute(
        predictions=formatted_predictions, references=references
    )
    return eval_results


def mnli_compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return load_metric("glue", "mnli").compute(
        predictions=predictions, references=labels
    )

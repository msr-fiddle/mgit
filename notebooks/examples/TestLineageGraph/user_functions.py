import numpy as np
from transformers import Trainer
from datasets import load_metric

def mnli_custom_test_function(model,lineage_dataset,tokenizer):
    dataset = lineage_dataset.get_dataset()
    feature_keys = lineage_dataset.feature_keys

    custom_label_map = {0:0,1:1,2:2}
    dataset_label2id = {"entailment":0,"neutral":1,"contradiction":2}
    if hasattr(model.config,"id2label"):
        id2label = model.config.id2label
        if set(dataset_label2id.keys()) == set([label_name.lower() for label_name in id2label.values()]):
            for k,v in id2label.items():
                custom_label_map[dataset_label2id[v.lower()]] = int(k)
    def label_swapping(example):
        example["label"] = custom_label_map[example["label"]]
        return example
    
    swap_dataset = dataset.map(label_swapping)

    def preprocess_function(examples):
        inputs = [examples[k] for k in feature_keys]
        return tokenizer(*inputs, truncation=True, max_length=128)
    
    tokenized_dataset = swap_dataset.map(
        preprocess_function, num_proc=4, batched=True,
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
        compute_metrics=mnli_compute_metrics
    )
    return trainer.evaluate()

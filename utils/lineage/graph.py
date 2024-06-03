import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import torch
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import transformers
import datasets
from tqdm import tqdm
import copy
import gc 

from pyvis.network import Network
from collections import defaultdict, OrderedDict, deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import misc_utils
import adapter
from model_utils import *
from ht.diffcheck import *
from ht.difflib import *
from lcs.diffcheck import lcs_one
from .delta_compression import *
from hashlib import sha256

__all__ = [
    "etypes",
    "trace_model",
    "divergence",
    "LineageNode",
    "LineageModel",
    "LineageGraph",
    "LineageDataset",
    "LineageTest",
    "LineageTrain",
    "get_all_ancestors",
    "get_all_offspring",
    "lineage_node_assertion",
    "ENTANGLE",
    "INIT_FROM",
    "FREEZE",
    "clear_torch_cache",
]
etypes = ["versioned", "adapted"]
device = torch.device("cpu")

#Accommodate the spelling issue in the config of Llama models we are using
exec(
    '''TOKENIZER_MAPPING_NAMES.update(llama=('LlamaTokenizer', 'LlamaTokenizerFast','LLaMATokenizer','LLaMATokenizerFast'))''',
     transformers.models.auto.tokenization_auto.__dict__
     )
transformers.models.llama.LLaMATokenizer = transformers.models.llama.tokenization_llama.LlamaTokenizer

def clear_torch_cache():
    gc.collect()
    torch.cuda.empty_cache()

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def trace_model(model, input_names=None, only_trace_base_mlm=True):
    if model.config.architectures and only_trace_base_mlm:
        if "formaskedlm" in model.config.architectures[0].casefold():
            model = model._modules[
                model.config.architectures[0]
                .casefold()
                .replace("formaskedlm", "")
            ]
    return find_submodules(model, input_names=input_names)

def divergence(node1, node2, input_names=None):
    """
    Computes the divergence score between the models represented by
    nodes node1 and node2.
    """
    if isinstance(input_names, tuple):
        (input_name1, input_name2) = input_names
    else:
        input_name1 = input_name2 = input_names

    # Extract the execution graph of each model
    if node1.traced is None or node1.is_retrace:
        if node1.is_retrace or node1.get_model().traced is None:
            node1.get_model().traced = trace_model(node1.get_model().model, input_name1)
        node1.traced = node1.get_model().traced
    _, _, namespace1, n1, e1 = node1.traced

    node2_prev_status = node2.is_model_loaded()
    if node2.traced is None or node2.is_retrace:
        if node2.is_retrace or node2.get_model().traced is None:
            node2.get_model().traced = trace_model(node2.get_model().model, input_name2)
        node2.traced = node2.get_model().traced
    _, _, namespace2, n2, e2 = node2.traced

    if  (not node2_prev_status) and node2.is_model_loaded():
        node2.unload_model(save_model=False)

    namespaces = [namespace1, namespace2]
    nodes = [n1, n2]
    edges = [e1, e2]
    # Structural match (partial equal on class type and layer attrs).
    node_matches, add_nodes, del_nodes, add_edges, del_edges = module_diff(
        namespaces, nodes, edges
    )

    structural_diff_edge = len(add_edges) + len(del_edges)
    # Contextual match.
    _, add_nodes, del_nodes, add_edges, del_edges = module_diff_contextual(
        namespaces,
        node_matches,
        edges,
        add_nodes,
        del_nodes,
        add_edges,
        del_edges,
    )
    contextual_diff_edge = len(add_edges) + len(del_edges)

    # Since diff does not guarantee the smallest diff, need to run twice
    # with m1 and m2 switched to get the 'min' diff.
    namespaces = [namespace2, namespace1]
    nodes = [n2, n1]
    edges = [e2, e1]
    # Structural match (partial equal on class type and layer attrs).
    node_matches, add_nodes, del_nodes, add_edges, del_edges = module_diff(
        namespaces, nodes, edges
    )

    structural_diff_edge = min(len(add_edges) + len(del_edges), structural_diff_edge)
    # Contextual match.
    _, add_nodes, del_nodes, add_edges, del_edges = module_diff_contextual(
        namespaces,
        node_matches,
        edges,
        add_nodes,
        del_nodes,
        add_edges,
        del_edges,
    )
    contextual_diff_edge = min(len(add_edges) + len(del_edges), contextual_diff_edge)

    total_edges = sum([len(edges) for _, edges in e1.items()]) + sum(
        [len(edges) for _, edges in e2.items()]
    )
    # Returned value can be a little bigger than 1 as the total number of edges does
    # not count the number of edges connecting to the pseudo input node.
    return (
        contextual_diff_edge / total_edges,
        structural_diff_edge / total_edges,
    )


class LineageDataset:
    """
    Wrapper class around datasets.
    """

    def __init__(self, *args, **kwargs):
        main_args = {"dataset", "feature_keys", "load_args", "load_kwargs_keys", "local_path"}
        allowed_preprocess_args = {
            "preprocess_function",
            "preprocess_function_path",
            "preprocess_function_name",
        }
        all_args = main_args.union(allowed_preprocess_args)
        self.__dict__.update((k, v) for k, v in kwargs.items())
        self.__dict__.update((k, None) for k in all_args if k not in kwargs.keys())

        if self.load_kwargs_keys is None:
            self.load_kwargs_keys = set([k for k in kwargs if k not in all_args])

        if self.load_args is None:
            self.load_args = args

        if self.local_path is not None:
            self.load_dataset = self.load_dataset_from_disk

        if self.preprocess_function is None:
            self.preprocess_function = misc_utils.try_get_function_from_file(
                path=self.preprocess_function_path,
                name=self.preprocess_function_name,
            )

    def load_from_file(file_path):
        with open(file_path, "r") as f:
            config = json.load(f)
        return LineageDataset.load_from_config(config)

    def load_from_config(config):
        return LineageDataset(**config)

    def to_json(self):
        res = dict(
            (k, v) for k, v in self.__dict__.items() if misc_utils.is_jsonable({k: v})
        )
        return res

    def get_dataset(self):
        if self.dataset is None:
            self.load_dataset()
            self.preprocess_dataset()
        return self.dataset

    def load_dataset(self):
        self.dataset = datasets.load_dataset(
            *self.load_args,
            **dict((k, self.__dict__[k]) for k in self.load_kwargs_keys),
        )

    def load_dataset_from_disk(self):
        self.dataset = datasets.load_from_disk(self.local_path)

    def preprocess_dataset(self):
        assert self.dataset is not None
        if self.preprocess_function is not None:
            self.dataset = self.preprocess_function(
                dataset=self.dataset, feature_keys=self.feature_keys
            )

    def concatenate(lineage_dataset_list, **kwargs):
        new_dataset = datasets.concatenate_datasets(
            [entry.get_dataset() for entry in lineage_dataset_list], **kwargs
        )
        return LineageDataset(
            dataset=new_dataset,
            feature_keys=lineage_dataset_list[0].feature_keys,
        )


class LineageTrain:
    """
    Wrapper class around training.
    """

    def __init__(self, **kwargs):
        allowed_main_args = {
            "node",
            "train_dataset",
            "eval_dataset",
            "step_counter",
            "data_iterator",
            "train_dataloader",
            "trainer",
        }
        allowed_dataset_args = {
            "eval_dataset",
            "eval_dataset_path",
            "train_dataset",
            "train_dataset_path",
        }
        allowed_perturb_args = {
            "perturbation_function_list",
            "perturbation_function_path_list",
            "perturbation_function_name_list",
        }
        self.allowed_trainer_args = {
            "data_collator",
            "compute_metrics",
            "callbacks",
        }  # TODO: data collator persistence.
        helper_trainer_args = {"compute_metrics_path", "compute_metrics_name"}
        allowed_preprocess_args = {
            "preprocess_function",
            "preprocess_function_path",
            "preprocess_function_name",
        }
        all_args = allowed_main_args.union(
            self.allowed_trainer_args,
            allowed_preprocess_args,
            helper_trainer_args,
            allowed_perturb_args,
            allowed_dataset_args,
        )
        self.allowed_training_args = set(kwargs.keys()) - all_args
        self.__dict__.update((k, v) for k, v in kwargs.items())
        self.__dict__.update((k, None) for k in all_args if k not in kwargs.keys())

        if self.train_dataset is None and self.train_dataset_path is not None:
            self.train_dataset = LineageDataset.load_from_file(self.train_dataset_path)

        if self.eval_dataset is None and self.eval_dataset_path is not None:
            self.eval_dataset = LineageDataset.load_from_file(self.eval_dataset_path)

        if self.node is not None:
            self.node.lineage_train = self

        if self.preprocess_function is None:
            self.preprocess_function = misc_utils.try_get_function_from_file(
                path=self.preprocess_function_path,
                name=self.preprocess_function_name,
            )

        if (
            self.perturbation_function_list is None
            and self.perturbation_function_path_list is not None
        ):
            assert len(self.perturbation_function_path_list) == len(
                self.perturbation_function_name_list
            )
            self.perturbation_function_list = []
            for i in range(len(self.perturbation_function_path_list)):
                perturb_func = misc_utils.try_get_function_from_file(
                    path=self.perturbation_function_path_list[i],
                    name=self.perturbation_function_name_list[i],
                )
                assert perturb_func is not None
                self.perturbation_function_list.append(perturb_func)

        if self.compute_metrics is None:
            self.compute_metrics = misc_utils.try_get_function_from_file(
                path=self.compute_metrics_path, name=self.compute_metrics_name
            )

        if self.step_counter is None:
            self.step_counter = 0

    def reset(self):
        self.step_counter = 0
        self.unload()

    def unload(self):
        if self.trainer is not None:
            del self.trainer
        self.trainer = None

    def load_from_file(file_path):
        with open(file_path, "r") as f:
            config = json.load(f)
        return LineageTrain.load_from_config(config)

    def load_from_config(config):
        live_attr = {"train_dataset", "eval_dataset"}
        if config is not None:
            if "train_dataset" in config:
                train_dataset = LineageDataset.load_from_config(config["train_dataset"])
            else:
                train_dataset = None
            if "eval_dataset" in config:
                eval_dataset = LineageDataset.load_from_config(config["eval_dataset"])
            else:
                eval_dataset = None
            persistent_args = dict(
                (k, v) for k, v in config.items() if k not in live_attr
            )
            lineage_train = LineageTrain(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                **persistent_args,
            )
            return lineage_train
        else:
            return None

    def to_json(self):
        live_attr = {"train_dataset", "eval_dataset"}
        res = dict(
            (k, v) for k, v in self.__dict__.items() if misc_utils.is_jsonable({k: v})
        )
        res.update(
            (k, self.__dict__[k].to_json())
            for k in live_attr
            if self.__dict__[k] is not None
        )
        return res

    def is_finished(self):
        if "num_train_epochs" not in self.__dict__:
            return True
        elif self.num_train_epochs == 0:
            return True
        else:
            train_loader = self.get_train_dataloader()
            return self.step_counter >= self.num_train_epochs * len(
                train_loader
            )

    def increment_step(self, num_steps=1):
        self.step_counter += num_steps

    def get_train_dataloader(self):
        if self.train_dataloader is None:
            self.train_dataloader = self.get_trainer().get_train_dataloader()
        return self.train_dataloader

    def next_data(self):
        """Get next data batch."""
        if self.data_iterator is None:
            self.data_iterator = enumerate(self.get_train_dataloader())
        step, inputs = next(self.data_iterator)
        if step + 1 >= len(self.get_train_dataloader()):
            self.data_iterator = enumerate(self.get_train_dataloader())
        return step, inputs

    def apply_perturbations(self, lineage_dataset):
        if self.perturbation_function_list is not None:
            p_dataset_list = []
            for p_func in self.perturbation_function_list:
                p_dataset = LineageDataset.load_from_config(lineage_dataset.__dict__)
                p_dataset.dataset = p_func(
                    p_dataset.get_dataset(), p_dataset.feature_keys
                )
                p_dataset_list.append(p_dataset)
            return LineageDataset.concatenate(
                [lineage_dataset] + p_dataset_list, axis=0
            )  # TODO improve this, possibly only concatenate datasets at the end after tokenization.
        else:
            return lineage_dataset

    def add_args_infer_collator(trainer_args, task_type, lineage_dataset, tokenizer):
        if "data_collator" not in trainer_args:
            if task_type == "sequence_classification":
                pass
            elif task_type == "question_answering":
                trainer_args["data_collator"] = transformers.default_data_collator
            elif task_type == "MaskedLM":
                trainer_args[
                    "data_collator"
                ] = transformers.DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm_probability=0.15
                )
            if "label" in lineage_dataset.get_dataset().features:
                pass
            elif "question" in lineage_dataset.get_dataset().features:
                trainer_args["data_collator"] = transformers.default_data_collator
            elif "text" in lineage_dataset.get_dataset().features:
                trainer_args[
                    "data_collator"
                ] = transformers.DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm_probability=0.15
                )

    def get_trainer(self, re_initialize=False):
        if self.trainer is not None and re_initialize is False:
            return self.trainer
        assert self.node is not None, "lineage train must have a node"
        model_inst = self.node.get_model(
            lineage_dataset=self.train_dataset, re_initialize=re_initialize
        )
        model = model_inst.model
        tokenizer = model_inst.tokenizer
        output_dir = self.node.output_dir

        local_trainer_args = {}

        LineageTrain.add_args_infer_collator(
            local_trainer_args,
            self.node.task_type,
            self.train_dataset,
            tokenizer,
        )

        train_dataset = self.apply_perturbations(self.train_dataset)
        if self.eval_dataset is not None:
            eval_dataset = self.apply_perturbations(self.eval_dataset)
        else:
            eval_dataset = None

        preprocess_function = self.preprocess_function
        if preprocess_function is not None and eval_dataset is not None:
            processed_train_dataset = preprocess_function(
                lineage_dataset=train_dataset, tokenizer=tokenizer, model=model
            )

            processed_eval_dataset = preprocess_function(
                lineage_dataset=eval_dataset, tokenizer=tokenizer, model=model
            )
            local_trainer_args["eval_dataset"] = processed_eval_dataset.get_dataset()
        elif preprocess_function is not None and eval_dataset is None:
            processed_train_dataset = preprocess_function(
                lineage_dataset=train_dataset, tokenizer=tokenizer, model=model
            )
            processed_eval_dataset = eval_dataset
        else:
            processed_train_dataset = train_dataset
            processed_eval_dataset = eval_dataset

        args = transformers.TrainingArguments(
            output_dir=os.path.join("tmp_trainer_args", output_dir),
            **dict(
                (k, self.__dict__[k])
                for k in self.allowed_training_args
                if self.__dict__[k] is not None
            ),
        )
        trainer = transformers.Trainer(
            model,
            args,
            train_dataset=processed_train_dataset.get_dataset(),
            tokenizer=tokenizer,
            **local_trainer_args,
            **dict(
                (k, self.__dict__[k])
                for k in self.allowed_trainer_args
                if self.__dict__[k] is not None
            ),
        )
        self.trainer = trainer
        return trainer

    def run(self, re_initialize):
        if re_initialize:
            self.step_counter = 0

        entangled_nodes = self.node.graph.entanglement_tracker.get_entangled(self.node)
        if len(entangled_nodes) == 1:
            # do vanilla training (no weight sharing)
            trainer = self.get_trainer(re_initialize)
            self.node.freeze_frozen_params()
            trainer.train()
            self.node.unfreeze_frozen_params()
            self.step_counter += self.num_train_epochs * len(
                self.get_train_dataloader()
            )
            self.node.get_model().model.to(device)
        else:
            active_set = entangled_nodes.union(set([self.node]))
            LineageTrain.MTL_train(active_set)
            for node in active_set:
                assert node.is_training_finished()
                node.get_model().model.to(device)

    def MTL_train(
        input_node_list,
        include_eval=None,
        task_importance=None,
        num_steps_per_epoch=None,
        num_train_epochs=None,
    ):
        node_list = list(input_node_list)
        trainer_list = [node.get_trainer() for node in node_list]
        dataloader_list = [
            entry.lineage_train.get_train_dataloader() for entry in node_list
        ]
        if task_importance is None:
            task_importance = [1.0 for entry in node_list]

        if include_eval is None:
            include_eval = [True for entry in node_list]

        if num_steps_per_epoch is None:
            num_steps_per_epoch = min([len(entry) for entry in dataloader_list])
        if num_train_epochs is None:
            num_train_epochs = max(
                [
                    1,
                    min([entry.args.num_train_epochs for entry in trainer_list]),
                ]
            )
        for i in range(len(trainer_list)):
            if trainer_list[i].optimizer is None:
                trainer_list[i].create_optimizer_and_scheduler(
                    num_training_steps=trainer_list[i].args.num_train_epochs
                    * len(dataloader_list[i])
                )

        for epoch in tqdm(range(num_train_epochs)):
            LineageTrain.inner_MTL_train(
                node_list=node_list,
                trainer_list=trainer_list,
                task_importance=task_importance,
                num_steps_per_epoch=num_steps_per_epoch,
            )

            # TODO remove save
            #for node in node_list:
            #    trainer = transformers.Trainer(
            #        node.get_model().model, tokenizer=node.get_model().tokenizer
            #    )
            #    trainer.save_model(node.output_dir + "-" + str(epoch))

        for node in node_list:
            node.lineage_train.increment_step(num_train_epochs * num_steps_per_epoch)

    def inner_MTL_train(node_list, trainer_list, task_importance, num_steps_per_epoch):
        for mtl_step in tqdm(range(num_steps_per_epoch)):
            for i in range(len(trainer_list)):
                step, inputs = node_list[i].lineage_train.next_data()
                trainer_list[i].model.train()
                inputs = inputs.to(trainer_list[i].model.device)
                cur_loss = task_importance[i] * trainer_list[i].compute_loss(
                    trainer_list[i].model, inputs
                )
                node_list[i].freeze_frozen_params()
                cur_loss.backward()
                trainer_list[i].optimizer.step()
                trainer_list[i].model.zero_grad()
                node_list[i].unfreeze_frozen_params()


class LineageTest:
    """
    Wrapper class around testing.
    """

    def __init__(self, **kwargs):
        allowed_main_args = {"name", "skip_test_for_compression"}
        allowed_dataset_args = {
            "eval_dataset",
            "eval_dataset_path",
        }
        allowed_test_function_args = {
            "custom_test_function",
            "custom_test_function_path",
            "custom_test_function_name",
        }
        allowed_postprocess_function_args = {
            "postprocess_function",
            "postprocess_function_path",
            "postprocess_function_name",
        }
        allowed_preprocess_args = {
            "preprocess_function",
            "preprocess_function_path",
            "preprocess_function_name",
        }
        allowed_perturb_args = {
            "perturbation_function",
            "perturbation_function_path",
            "perturbation_function_name",
        }
        allowed_test_success_args = {
            "test_success_condition",
            "test_success_condition_path",
            "test_success_condition_name",
        }
        self.allowed_training_args = {"metric_for_best_model"}
        self.allowed_trainer_args = {
            "data_collator",
            "compute_metrics",
        }  # TODO: data_collator persistence.
        helper_trainer_args = {"compute_metrics_path", "compute_metrics_name"}
        all_args = allowed_main_args.union(
            allowed_dataset_args,
            allowed_test_function_args,
            self.allowed_trainer_args,
            self.allowed_training_args,
            allowed_preprocess_args,
            allowed_test_success_args,
            allowed_perturb_args,
            allowed_postprocess_function_args,
            helper_trainer_args,
        )
        assert not set([k for k in kwargs]).difference(all_args), set(
            [k for k in kwargs]
        ).difference(all_args)
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in all_args)
        self.__dict__.update((k, None) for k in all_args if k not in kwargs.keys())

        if self.eval_dataset is None and self.eval_dataset_path is not None:
            self.eval_dataset = LineageDataset.load_from_file(self.eval_dataset_path)

        if self.custom_test_function is None:
            self.custom_test_function = misc_utils.try_get_function_from_file(
                path=self.custom_test_function_path,
                name=self.custom_test_function_name,
            )

        if self.perturbation_function is None:
            self.perturbation_function = misc_utils.try_get_function_from_file(
                path=self.perturbation_function_path,
                name=self.perturbation_function_name,
            )

        if self.preprocess_function is None:
            self.preprocess_function = misc_utils.try_get_function_from_file(
                path=self.preprocess_function_path,
                name=self.preprocess_function_name,
            )

        if self.postprocess_function is None:
            self.postprocess_function = misc_utils.try_get_function_from_file(
                path=self.postprocess_function_path, name=self.postprocess_function_name
            )

        if self.compute_metrics is None:
            self.compute_metrics = misc_utils.try_get_function_from_file(
                path=self.compute_metrics_path, name=self.compute_metrics_name
            )

        if self.test_success_condition is None:
            self.test_success_condition = misc_utils.try_get_function_from_file(
                path=self.test_success_condition_path,
                name=self.test_success_condition_name,
            )

        assert isinstance(self.name, str)

        if self.skip_test_for_compression is None:
            self.skip_test_for_compression = False
        assert isinstance(self.skip_test_for_compression, bool)

    def load_from_file(file_path):
        with open(file_path, "r") as f:
            config = json.load(f)
        return LineageTest.load_from_config(config)

    def load_from_config(config):
        live_attr = {"eval_dataset"}
        if "eval_dataset" in config:
            eval_dataset = LineageDataset.load_from_config(config["eval_dataset"])
        else:
            eval_dataset = None
        persistent_args = dict((k, v) for k, v in config.items() if k not in live_attr)
        return LineageTest(**persistent_args, eval_dataset=eval_dataset)

    def to_json(self):
        live_attr = {"eval_dataset"}
        res = dict(
            (k, v) for k, v in self.__dict__.items() if misc_utils.is_jsonable({k: v})
        )
        res.update(
            (k, self.__dict__[k].to_json())
            for k in live_attr
            if self.__dict__[k] is not None
        )
        return res

    def run(self, node, return_results):
        assert self.eval_dataset is not None
        model_inst = node.get_model(
            lineage_dataset=self.eval_dataset, re_initialize=False
        )
        model = model_inst.model
        tokenizer = model_inst.tokenizer
        output_dir = node.output_dir

        if self.custom_test_function is not None:  # TODO rename
            eval_results = self.custom_test_function(
                model=model,
                lineage_dataset=self.eval_dataset,
                tokenizer=tokenizer,
            )

            if self.test_success_condition is not None:
                test_result = bool(self.test_success_condition(eval_results))
            else:
                test_result = None

            return test_result, eval_results if return_results else test_result

        local_trainer_args = {}
        LineageTrain.add_args_infer_collator(
            local_trainer_args, node.task_type, self.eval_dataset, tokenizer
        )

        if self.perturbation_function is not None:
            eval_dataset = LineageDataset.load_from_config(self.eval_dataset.__dict__)
            eval_dataset.dataset = self.perturbation_function(
                eval_dataset.get_dataset(), eval_dataset.feature_keys
            )
        else:
            eval_dataset = self.eval_dataset

        if self.preprocess_function is not None:
            processed_eval_dataset = self.preprocess_function(
                lineage_dataset=eval_dataset, tokenizer=tokenizer, model=model
            )
        else:
            processed_eval_dataset = eval_dataset

        args = transformers.TrainingArguments(
            output_dir=os.path.join("tmp_trainer_args", output_dir),
            **dict(
                (k, self.__dict__[k])
                for k in self.allowed_training_args
                if self.__dict__[k] is not None
            ),
        )
        trainer = transformers.Trainer(
            model,
            args,
            tokenizer=tokenizer,
            eval_dataset=processed_eval_dataset.get_dataset(),
            **dict(
                (k, self.__dict__[k])
                for k in self.allowed_trainer_args
                if self.__dict__[k] is not None
            ),
            **local_trainer_args,
        )
        if self.postprocess_function is not None:
            raw_predictions = trainer.predict(processed_eval_dataset.get_dataset())
            eval_results = self.postprocess_function(
                raw_predictions=raw_predictions,
                processed_lineage_dataset=processed_eval_dataset,
                lineage_dataset=eval_dataset,
                tokenizer=tokenizer,
            )
        else:
            eval_results = trainer.evaluate()

        if self.test_success_condition is not None:
            test_result = bool(self.test_success_condition(eval_results))
        else:
            test_result = None

        return test_result, eval_results if return_results else test_result


class LineageModel:
    def __init__(self, node, init_checkpoint, lineage_dataset, re_initialize):
        """
        instantiates a LineageModel and loads in the pytorch model

        Args:
            node (LineageNode): lineage node corresponding to this model
            init_checkpoint (string): file path to initial checkpoint of this model
            lineage_dataset (None or LineageDataset): optional lineage dataset used to infer the architecture type (e.g. sequence classification vs. MLM)
            re_initialize (bool): whether to load in existing stored weights of model or re-initialize to initial value
        """
        self.node = node
        self.model = None
        self.tokenizer = None
        self.traced = None
        self.initialize(
            init_checkpoint=init_checkpoint,
            output_dir=node.output_dir,
            lineage_dataset=lineage_dataset,
            re_initialize=re_initialize,
        )

    def set_state_reference(self, name, src_obj):
        cur_dict = self.model
        key_list = name.split(".")
        for key in key_list[:-1]:
            cur_dict = cur_dict._modules[key]
        if key_list[-1] in cur_dict._parameters:
            cur_dict._parameters[key_list[-1]] = src_obj
        elif key_list[-1] in cur_dict._buffers:
            cur_dict._buffers[key_list[-1]] = src_obj
        elif key_list[-1] in cur_dict._modules:
            cur_dict._modules[key_list[-1]] = src_obj
        else:
            assert False, "could not find " + name

    def get_state_reference(self, name):
        cur_dict = self.model
        key_list = name.split(".")
        for key in key_list[:-1]:
            cur_dict = cur_dict._modules[key]
        if key_list[-1] in cur_dict._parameters:
            return cur_dict._parameters[key_list[-1]]
        elif key_list[-1] in cur_dict._buffers:
            return cur_dict._buffers[key_list[-1]]
        elif key_list[-1] in cur_dict._modules:
            return cur_dict._modules[key_list[-1]]
        else:
            assert False, "no state reference found for " + name

    def save(self, vanilla_save=True, unique_hash=False):
        if self.node.is_dirty:
            if self.node.init_checkpoint == self.node.output_dir:
                print(
                    "WARNING: saving a model which has same init_checkpoint as output_dir"
                )
            assert self.model is not None
            assert self.tokenizer is not None
            # args = transformers.TrainingArguments(output_dir=self.output_dir, no_cuda=True)
            if vanilla_save:
                if self.node.model_type != "torchvision":
                    trainer = transformers.Trainer(self.model, tokenizer=self.tokenizer)
                    trainer.save_model(self.node.output_dir)
                    self.model.to(device)
                else:
                    if not os.path.exists(self.node.output_dir):
                        os.mkdir(self.node.output_dir)
                    torch.save(
                        self.model,
                        os.path.join(self.node.output_dir, "pytorch_model.bin"),
                    )
            self.save_custom_state_dict(unique_hash=unique_hash)
            if self.traced:
                torch.save(self.traced, os.path.join(self.node.output_dir, "traced.pt"))
            self.node.is_dirty = False

    def model_exists(model_checkpoint, trust_remote_code=False):
        """
        Indicates whether a valid lineage model is stored by checking if the config of the model can be loaded in

        Args:
            model_checkpoint (string): file path of checkpoint to verify
        """
        try:
            transformers.AutoConfig.from_pretrained(model_checkpoint, trust_remote_code=trust_remote_code)
            return True
        except Exception as e:
            print(str(e))
            return False

    def is_custom_dict_available(self):
        """
        checks if parameters have been registered and saved to global parameter store
        """
        return (
            self.node.graph is not None
            and len(self.node.local_to_global_id) > 0
            and self.node.graph.entanglement_tracker.has_global_ids(
                self.node.local_to_global_id.values()
            )
        )

    def load_custom_state_dict(self, dtype=None):
        """
        loads state_dict from global parameter store
        """
        global_state_dict = self.node.graph.entanglement_tracker.load_global_dict(
            self.node.local_to_global_id.values()
        )
        if dtype is not None:
            local_state_dict = dict(
                (k, global_state_dict[self.node.local_to_global_id[k]].to(dtype))
                for k in self.node.local_to_global_id
            )
        else:
            local_state_dict = dict(
                (k, global_state_dict[self.node.local_to_global_id[k]])
                for k in self.node.local_to_global_id
            )
        return local_state_dict

    def compress(self, compression_ratio_threshold):
        v2 = {name: weights for name, weights in self.model.state_dict().items() 
              if self.model.state_dict()[name].dtype != torch.bool}
        v2_keys = list(v2.keys())
        # This happens in bn layers among torchvision models
        [v2.pop(k) for k in v2_keys if "num_batches_tracked" in k]
        v1 = {name: torch.zeros(weights.shape, dtype=weights.dtype) for name, weights in v2.items()}

        if self.node.graph.compression_mode == "rle":
            compressed, ratio = delta_compress_rle(
                v2, v1, quantize_delta=self.node.quantize_delta
            )
            if ratio < compression_ratio_threshold:
                return False, {}, {}
            decompressed = delta_decompress_rle(
                compressed, v1, quantize_delta=self.node.quantize_delta
            )
        elif self.node.graph.compression_mode == "lzma":
            compressed, ratio = delta_compress_lzma(
                v2, v1, quantize_delta=self.node.quantize_delta
            )
            if ratio < compression_ratio_threshold:
                return False, {}, {}
            decompressed = delta_decompress_lzma(
                compressed, v1, quantize_delta=self.node.quantize_delta
            )
        elif self.node.graph.compression_mode == "sparse":
            compressed, ratio = delta_compress_sparse(
                v2, v1, quantize_delta=self.node.quantize_delta
            )
            if ratio < compression_ratio_threshold:
                return False, {}, {}
            decompressed = delta_decompress_sparse(
                compressed, v1, quantize_delta=self.node.quantize_delta
            )
        elif self.node.graph.compression_mode == "dict":
            compressed, ratio = delta_compress_dict(
                v2, v1, quantize_delta=self.node.quantize_delta
            )
            if ratio < compression_ratio_threshold:
                return False, {}, {}
            decompressed = delta_decompress_dict(
                compressed, v1, quantize_delta=self.node.quantize_delta
            )
        elif self.node.graph.compression_mode == "sparse_dict":
            compressed, ratio = delta_compress_sparse_dict(
                v2, v1, quantize_delta=self.node.quantize_delta
            )
            if ratio < compression_ratio_threshold:
                return False, {}, {}
            decompressed = delta_decompress_sparse_dict(
                compressed, v1, quantize_delta=self.node.quantize_delta
            )
        else:
            compressed, ratio = delta_compress(
                v2, v1, quantize_delta=self.node.quantize_delta
            )
            if ratio < compression_ratio_threshold:
                return False, {}, {}
            decompressed = delta_decompress(
                compressed, v1, quantize_delta=self.node.quantize_delta
            )

        decompressed_state_dict = self.model.state_dict()
        for layer, weights in decompressed.items():
            decompressed_state_dict[layer] = weights
        return True, compressed, decompressed_state_dict

    def delta_compress(self, parent, compression_ratio_threshold):
        def equal(x, y):
            return x[1] == y[1]

        parent_layers = list(parent.state_dict().keys())
        model_layers = list(self.model.state_dict().keys())
        a = [(name, parent.state_dict()[name].shape) for name in parent_layers]
        b = [(name, self.model.state_dict()[name].shape) for name in model_layers]
        name_map = {
            model_layers[p[1]]: parent_layers[p[0]] for p in lcs_one(a, b, equal)
        }
        name_map_keys = list(name_map.keys())
        # This happens in bn layers among torchvision models
        [name_map.pop(k) for k in name_map_keys if "num_batches_tracked" in k]
        v1 = {
            name: weights
            for name, weights in parent.state_dict().items()
            if name in name_map.values() and parent.state_dict()[name].dtype != torch.bool
        }
        v2 = {
            name: weights
            for name, weights in self.model.state_dict().items()
            if name in name_map.keys() and self.model.state_dict()[name].dtype != torch.bool
        }

        if self.node.graph.compression_mode == "rle":
            compressed, ratio = delta_compress_rle(
                v2, v1, name_map=name_map, quantize_delta=self.node.quantize_delta
            )
            if ratio < compression_ratio_threshold:
                return False, {}, {}
            decompressed = delta_decompress_rle(
                compressed,
                v1,
                name_map=name_map,
                quantize_delta=self.node.quantize_delta,
            )
        elif self.node.graph.compression_mode == "lzma":
            compressed, ratio = delta_compress_lzma(
                v2, v1, name_map=name_map, quantize_delta=self.node.quantize_delta
            )
            if ratio < compression_ratio_threshold:
                return False, {}, {}
            decompressed = delta_decompress_lzma(
                compressed,
                v1,
                name_map=name_map,
                quantize_delta=self.node.quantize_delta,
            )
        elif self.node.graph.compression_mode == "sparse":
            compressed, ratio = delta_compress_sparse(
                v2, v1, name_map=name_map, quantize_delta=self.node.quantize_delta
            )
            if ratio < compression_ratio_threshold:
                return False, {}, {}
            decompressed = delta_decompress_sparse(
                compressed,
                v1,
                name_map=name_map,
                quantize_delta=self.node.quantize_delta,
            )
        elif self.node.graph.compression_mode == "dict":
            compressed, ratio = delta_compress_dict(
                v2, v1, name_map=name_map, quantize_delta=self.node.quantize_delta
            )
            if ratio < compression_ratio_threshold:
                return False, {}, {}
            decompressed = delta_decompress_dict(
                compressed,
                v1,
                name_map=name_map,
                quantize_delta=self.node.quantize_delta,
            )
        elif self.node.graph.compression_mode == "sparse_dict":
            compressed, ratio = delta_compress_sparse_dict(
                v2, v1, name_map=name_map, quantize_delta=self.node.quantize_delta
            )
            if ratio < compression_ratio_threshold:
                return False, {}, {}
            decompressed = delta_decompress_sparse_dict(
                compressed,
                v1,
                name_map=name_map,
                quantize_delta=self.node.quantize_delta,
            )
        else:
            compressed, ratio = delta_compress(
                v2, v1, name_map=name_map, quantize_delta=self.node.quantize_delta
            )
            if ratio < compression_ratio_threshold:
                return False, {}, {}
            decompressed = delta_decompress(
                compressed,
                v1,
                name_map=name_map,
                quantize_delta=self.node.quantize_delta,
            )

        decompressed_state_dict = self.model.state_dict()
        for layer, weights in decompressed.items():
            decompressed_state_dict[layer] = weights
        return True, compressed, decompressed_state_dict

    def is_delta_compressible(self, accuracy_threshold, compression_ratio_threshold):
        parents = self.node.get_parents("adapted")
        if len(parents) == 0:
            return False, {}, {}

        self.node.run_all_tests()
        model_copy = copy.deepcopy(self.model)
        results_before_compression = copy.deepcopy(self.node.test_result_dict)

        # Delta compress with first parent if a node has multiple parents.
        # TODO: Do something smarter?
        (
            is_delta_compressible_flag,
            compressed,
            decompressed_state_dict,
        ) = self.delta_compress(
            parents[0].get_model().model, compression_ratio_threshold
        )

        if not is_delta_compressible_flag:
            del self.model
            self.model = model_copy.to(device)
            return False, {}, {}

        self.model.load_state_dict(decompressed_state_dict)
        self.model.to(device)
        
        self.node.run_all_tests()
        results_after_compression = self.node.test_result_dict
        for test_name, test_result in results_before_compression.items():
            if self.node.graph.test_dict[test_name].skip_test_for_compression:
                continue
            metric_name = self.node.graph.test_dict[test_name].metric_for_best_model
            if metric_name not in test_result["results"]:
                metric_name = "eval_" + metric_name
            if (
                (
                    test_result["results"][metric_name]
                    - results_after_compression[test_name]["results"][metric_name]
                )
                / abs(test_result["results"][metric_name])
                > accuracy_threshold
            ):
                print(
                    f'WARNING: compression failed because before compression: {metric_name}: {test_result["results"][metric_name]} and after compression:\
                      {metric_name}: {results_after_compression[test_name]["results"][metric_name]}'
                )
                del self.model
                self.model = model_copy.to(device)
                self.node.test_result_dict = results_before_compression
                return False, {}, {}

        if len(compressed) > 0:
            register_init_from(parents[0], self.node)
            self.node.graph.entanglement_tracker.temporary_register_state(self.node)
            self.node.is_dirty = True
            return True, compressed, decompressed_state_dict
        else:
            return False, {}, {}

    def is_compressible(self, accuracy_threshold, compression_ratio_threshold):
        
        self.node.run_all_tests()
        model_copy = copy.deepcopy(self.model)
        results_before_compression = copy.deepcopy(self.node.test_result_dict)

        is_compressible_flag, compressed, decompressed_state_dict = self.compress(
            compression_ratio_threshold
        )

        if not is_compressible_flag:
            self.model = model_copy
            return False, {}, {}

        self.model.load_state_dict(decompressed_state_dict)
        self.node.run_all_tests()
        results_after_compression = self.node.test_result_dict
        for test_name, test_result in results_before_compression.items():
            if self.node.graph.test_dict[test_name].skip_test_for_compression:
                continue
            metric_name = self.node.graph.test_dict[test_name].metric_for_best_model
            if metric_name not in test_result["results"]:
                metric_name = "eval_" + metric_name
            if (
                test_result["results"][metric_name] == 0
                or (
                    test_result["results"][metric_name]
                    - results_after_compression[test_name]["results"][metric_name]
                )
                / test_result["results"][metric_name]
                > accuracy_threshold
            ):
                print(
                    f'WARNING: compression failed because {metric_name}: {results_after_compression[test_name]["results"][metric_name]}'
                )
                self.model = model_copy
                self.node.test_result_dict = results_before_compression
                return False, {}, {}

        if len(compressed) > 0:
            self.node.graph.entanglement_tracker.temporary_register_state(self.node)
            self.node.is_dirty = True
            return True, compressed, decompressed_state_dict
        else:
            return False, {}, {}

    def save_custom_state_dict(
        self, accuracy_threshold=0.02, compression_ratio_threshold=1, unique_hash=False
    ):
        """
        stores state_dict to global parameter store
        """
        if self.node == None or self.node.graph == None:
            print("WARNING: model has to be loaded into graph to save custom_state_dict!")
            return

        if self.node.graph.single_model_compression:
            is_compressible, compressed, decompressed = self.is_compressible(
                accuracy_threshold, compression_ratio_threshold
            )
            print("is_compressible:", is_compressible)
            for k, v in self.model.state_dict().items():
                if is_compressible and k in compressed:
                    self.node.graph.entanglement_tracker.update_delta_store(
                        self.node.local_to_global_id[k],
                        compressed[k],
                        decompressed[k],
                        self.node.quantize_delta,
                        node_name=self.node.output_dir if unique_hash else None
                    )
                else:
                    self.node.graph.entanglement_tracker.update_param_store(
                        self.node.local_to_global_id[k], v,
                        node_name=self.node.output_dir if unique_hash else None
                    )
        else:
            if self.node.is_delta:
                (
                    is_delta_compressible,
                    compressed,
                    decompressed,
                ) = self.is_delta_compressible(
                    accuracy_threshold, compression_ratio_threshold
                )
            else:
                is_delta_compressible = self.node.is_delta
            #print("is_delta_compressible:", is_delta_compressible)

            for k, v in self.model.state_dict().items():
                if is_delta_compressible and k in compressed:
                    self.node.graph.entanglement_tracker.update_delta_store(
                        self.node.local_to_global_id[k],
                        compressed[k],
                        decompressed[k],
                        self.node.quantize_delta,
                        node_name=self.node.output_dir if unique_hash else None
                    )
                else:
                    self.node.graph.entanglement_tracker.update_param_store(
                        self.node.local_to_global_id[k], v,
                        node_name=self.node.output_dir if unique_hash else None
                    )
        clear_torch_cache()            

    def initialize(self, init_checkpoint, output_dir, lineage_dataset, re_initialize):
        """
        internal helper function to load in the pytorch model.
        If checkpoint does not exist at output_dir or re_initialize is True, model is loaded from init_checkpoint, otherwise model is
        loaded from output_dir.

        If node.task_type is specifised, model is loaded according to task type
        Otherwise if node has training dataset associated with it, then the task type is inferred according to the feature names of the dataset

        If node.task_type is not specified and task type cannot be inferreed by dataset, then model is loaded according to the config of the checkpoint

        Args:
            init_checkpoint (None or string): file path to initial checkpoint of this model
            output_dir (string): file path (and node id) to model
            lineage_dataset (None or LineageDataset): optional lineage dataset used to infer the architecture type (e.g. sequence classification vs. MLM)
            re_initialize (bool): whether to load in existing stored weights of model or re-initialize to initial value
        """
        if self.node.model_type == "torchvision":
            self.tokenizer = lambda x: x
            if output_dir and os.path.exists(
                os.path.join(output_dir, "pytorch_model.bin")
            ):
                model_checkpoint = output_dir
            elif init_checkpoint and os.path.exists(
                os.path.join(init_checkpoint, "pytorch_model.bin")
            ):
                model_checkpoint = init_checkpoint
            else:
                print("WARNING: checkpoint does not exist!")
                return

            print("loading model:", model_checkpoint)
            self.model = torch.load(
                                    os.path.join(model_checkpoint, "pytorch_model.bin"), 
                                    map_location = device,
                                    )
            if self.is_custom_dict_available():
                custom_state_dict = self.load_custom_state_dict(dtype=self.model.dtype)
                if custom_state_dict is not None:
                    self.model.load_state_dict(custom_state_dict)
            return

        elif LineageModel.model_exists(output_dir, self.node.trust_remote_code) and (re_initialize is False):
            model_checkpoint = output_dir
        elif LineageModel.model_exists(init_checkpoint, self.node.trust_remote_code):
            model_checkpoint = init_checkpoint
        else:
            print("WARNING: checkpoint does not exist!")
            return

        print("loading model:", model_checkpoint)

        if self.is_custom_dict_available():
            custom_state_dict = self.load_custom_state_dict()
            assert custom_state_dict is not None
        else:
            custom_state_dict = None

        model = None
        try:
            if lineage_dataset is not None and self.node.task_type is None:
                # infer task type by lineage_dataset
                print("attempting load model by infering task type")
                model = self.load_model_infer_data(
                    model_checkpoint,
                    lineage_dataset,
                    custom_state_dict=custom_state_dict,
                )
            elif self.node.task_type is not None:
                # load model according to task type label
                print("attempting to load model by specified task type")
                model = self.load_model_by_task_type(
                    model_checkpoint,
                    self.node.task_type,
                    lineage_dataset=lineage_dataset,
                    custom_state_dict=custom_state_dict,
                )
            if model is None:
                print("attempting to load model by config")
                model = self.load_model_by_config(
                    model_checkpoint, custom_state_dict=custom_state_dict
                )
        except Exception as e:
            print("could not automatically load model!")
            print(str(e))

        if model is not None:
            model = self.check_adapter(model, model_checkpoint)
        
        self.model = model
        self.tokenizer = self.load_tokenizer(model_checkpoint)

        if os.path.exists(os.path.join(model_checkpoint, "traced.pt")):
            self.traced = torch.load(
                                    os.path.join(model_checkpoint, "traced.pt"), 
                                    map_location = torch.device('cpu'),
                                    )

    # TODO structural adaptation should be able to handle generic user-defined transformations
    def check_adapter(self, model, model_checkpoint):
        """
        modifies architecture for special adapter architecture.
        """
        if hasattr(model.config, "adapter_size"):
            adapter.add_adapters(model, model.config)
            model.load_state_dict(
                torch.load(
                    model_checkpoint + "/pytorch_model.bin",
                    map_location = model.device,
                )
            )
        return model

    def load_model_by_task_type(
        self, model_checkpoint, task_type, lineage_dataset=None, custom_state_dict=None
    ):
        """
        attempt to load model by task type label. If task type is not recognized, returns None

        Returns:
            None or LineageModel
        """
        if "sequence_classification" == task_type:
            if (
                lineage_dataset is not None
                and hasattr(lineage_dataset.get_dataset(), 'features')
                    and "num_classes" in lineage_dataset.get_dataset().features["label"].__dict__
            ):
                return transformers.AutoModelForSequenceClassification.from_pretrained(
                    model_checkpoint,
                    state_dict=custom_state_dict,
                    num_labels=lineage_dataset.get_dataset()
                    .features["label"]
                    .num_classes,
                    trust_remote_code=self.node.trust_remote_code,
                )

            elif LineageModel.model_exists(model_checkpoint, self.node.trust_remote_code):
                config = transformers.AutoConfig.from_pretrained(
                    model_checkpoint,
                    trust_remote_code=self.node.trust_remote_code,
                )
                if (
                    config.architectures is not None
                    and "SequenceClassification" in config.architectures[0]
                ):
                    return (
                        transformers.AutoModelForSequenceClassification.from_pretrained(
                            model_checkpoint,
                            state_dict=custom_state_dict,
                            trust_remote_code=self.node.trust_remote_code,
                        )
                    )

            else:
                return transformers.AutoModelForSequenceClassification.from_pretrained(
                    model_checkpoint,
                    state_dict=custom_state_dict,
                    num_labels=1,
                    trust_remote_code=self.node.trust_remote_code,
                )
        elif "question_answering" == task_type:
            return transformers.AutoModelForQuestionAnswering.from_pretrained(
                model_checkpoint,
                state_dict=custom_state_dict,
                trust_remote_code=self.node.trust_remote_code,
            )
        elif "MaskedLM" == task_type:
            return transformers.AutoModelForMaskedLM.from_pretrained(
                model_checkpoint,
                state_dict=custom_state_dict,
                trust_remote_code=self.node.trust_remote_code,
            )
        elif "seq2seq" == task_type:
            return transformers.AutoModelForSeq2SeqLM.from_pretrained(
                model_checkpoint,
                state_dict=custom_state_dict,
                trust_remote_code=self.node.trust_remote_code,
            )
        elif "causallm" == task_type:
            print("WARNING: AutoModelForCausalLM tries to load model in float16")
            return transformers.AutoModelForCausalLM.from_pretrained(
                model_checkpoint,
                state_dict=custom_state_dict,
                torch_dtype=torch.float16,#TODO: remove constraint on dtype
                low_cpu_mem_usage=True,
                trust_remote_code=self.node.trust_remote_code,
            )
        else:
            return None

    def load_model_infer_data(
        self, model_checkpoint, lineage_dataset, custom_state_dict=None
    ):
        """
        attempt to infer task type by feature names of the dataset. If cannot infer task type, returns None

        Returns:
            None or LineageModel
        """
        if "label" in lineage_dataset.get_dataset().features:
            if (
                "num_classes"
                in lineage_dataset.get_dataset().features["label"].__dict__
            ):
                return transformers.AutoModelForSequenceClassification.from_pretrained(
                    model_checkpoint,
                    state_dict=custom_state_dict,
                    num_labels=lineage_dataset.get_dataset()
                    .features["label"]
                    .num_classes,
                    trust_remote_code=self.node.trust_remote_code,
                    
                )
            else:
                return transformers.AutoModelForSequenceClassification.from_pretrained(
                    model_checkpoint,
                    state_dict=custom_state_dict,
                    num_labels=1,
                    trust_remote_code=self.node.trust_remote_code,
                )
        elif "question" in lineage_dataset.get_dataset().features:
            return transformers.AutoModelForQuestionAnswering.from_pretrained(
                model_checkpoint,
                state_dict=custom_state_dict,
                trust_remote_code=self.node.trust_remote_code,
            )
        elif "text" in lineage_dataset.get_dataset().features:
            return transformers.AutoModelForMaskedLM.from_pretrained(
                model_checkpoint,
                state_dict=custom_state_dict,
                trust_remote_code=self.node.trust_remote_code,
            )
        else:
            return None

    def load_model_by_config(self, model_checkpoint, custom_state_dict=None):
        config = transformers.AutoConfig.from_pretrained(
            model_checkpoint,
            trust_remote_code=self.node.trust_remote_code,
        )
        if config.architectures is not None:
            architecture = config.architectures[0]
            model = getattr(transformers, architecture).from_pretrained(
                model_checkpoint, 
                state_dict=custom_state_dict,
                trust_remote_code=self.node.trust_remote_code,
            )
        else:
            # TODO this should be automodel
            # model = transformers.AutoModelForMaskedLM.from_pretrained(model_checkpoint, state_dict=custom_state_dict)
            model = transformers.AutoModel.from_pretrained(
                model_checkpoint, 
                state_dict=custom_state_dict,
                trust_remote_code=self.node.trust_remote_code,
            )
        return model

    def load_tokenizer(self, model_checkpoint):
        found_tokenizer = False
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_checkpoint,
                trust_remote_code=self.node.trust_remote_code,
            )
            found_tokenizer = True
        except Exception as e:
            print(e)

        if not found_tokenizer:
            try:
                tokenizer = transformers.AutoFeatureExtractor.from_pretrained(
                    model_checkpoint,
                    trust_remote_code=self.node.trust_remote_code,
                )
                found_tokenizer = True
            except Exception as e:
                print(e)

        if not found_tokenizer:
            print("could not load tokenizer or extractor!")
            tokenizer = None
        return tokenizer

    def __del__(self):
        if self.model is not None:
            del self.model


class LineageNode:
    """
    Represent a node in the lineaga graph.
    """

    def __init__(self, output_dir, **kwargs):
        attr_args = {
            "graph",
            "parent_dict",
            "children_dict",
            "init_checkpoint",
            "local_to_global_id",
            "frozen_local_ids",
            "model_inst",
            "model_type",
            "task_type",
            "traced",
            "quantize_delta",
            "is_retrace",
            "is_dirty",
            "is_delta",
            "is_init",
            "is_unload",#Deprecated
            "trust_remote_code",
        }
        allowed_test_args = {
            "test_result_dict",
            "test_name_list",
        }
        allowed_lineage_train_args = {
            "lineage_train",
            "lineage_train_path",
        }
        allowed_init_function_args = {
            "model_init_function",
            "model_init_function_path",
            "model_init_function_name",
        }
        all_args = attr_args.union(
            allowed_test_args,
            allowed_lineage_train_args,
            allowed_init_function_args,
        )
        illegal_args = set(kwargs.keys()) - all_args
        assert not illegal_args, illegal_args
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in all_args)
        self.__dict__.update((k, None) for k in all_args if k not in kwargs.keys())

        self.output_dir = output_dir
        assert isinstance(self.output_dir, str)

        if self.trust_remote_code is None:
            self.trust_remote_code = False

        if self.is_dirty is None:
            self.is_dirty = True

        if self.is_init is None:
            self.is_init = False

        if self.is_delta is None:
            self.is_delta = False

        if self.is_retrace is None:
            self.is_retrace = False

        if self.quantize_delta is None:
            self.quantize_delta = True

        if self.parent_dict is None:
            self.parent_dict = defaultdict(list)
        else:
            self.parent_dict = defaultdict(list, self.parent_dict.items())

        if self.children_dict is None:
            self.children_dict = defaultdict(list)
        else:
            self.children_dict = defaultdict(list, self.children_dict.items())

        if self.test_result_dict is None:
            self.test_result_dict = {}

        if self.test_name_list is None:
            self.test_name_list = []
        else:
            for test_name in self.test_name_list:
                if test_name not in self.test_result_dict:
                    self.test_result_dict[test_name] = {
                        "results": None,
                        "success": None,
                    }

        if self.lineage_train is None and self.lineage_train_path is not None:
            self.lineage_train = LineageTrain.load_from_file(self.lineage_train_path)

        if self.lineage_train is not None:
            self.lineage_train.node = self

        if self.local_to_global_id is None:
            self.local_to_global_id = {}

        if self.frozen_local_ids is None:
            self.frozen_local_ids = set()
        else:
            self.frozen_local_ids = set(self.frozen_local_ids)

        if self.model_init_function is None:
            self.model_init_function = misc_utils.try_get_function_from_file(
                path=self.model_init_function_path,
                name=self.model_init_function_name,
            )

    # ===============================================================================
    # Model freezing and unfreezing helper methods.
    # ===============================================================================

    def copy(self, new_node_name):
        new_node_config = self.to_json()
        new_node_config["output_dir"] = new_node_name
        new_node = LineageNode.load_from_config(new_node_config)

        new_node.is_dirty = True
        new_node.is_init = False
        new_node.traced = copy.deepcopy(self.traced)

        new_node.parent_dict = defaultdict(list)
        new_node.children_dict = defaultdict(list)

        new_node.test_result_dict = {}
        for test_name in new_node.test_name_list:
            new_node.test_result_dict[test_name] = {
                "results": None,
                "success": None,
            }

        if new_node.lineage_train is not None:
            new_node.lineage_train.reset()

        new_node.local_to_global_id = {}

        new_node.frozen_local_ids = set()

        return new_node

    def has_compressed(self):
        for global_id in self.local_to_global_id.values():
            file_path = os.path.join(
                self.graph.param_home, self.graph.entanglement_tracker.global_to_hash[global_id]
            )
            tmp_hash_dict = torch.load(file_path, map_location = torch.device('cpu'))

            if tmp_hash_dict["is_delta"]:
                return True
        return False

    def freeze_frozen_params(self):
        for local_name in self.frozen_local_ids:
            self.get_model().get_state_reference(local_name).requires_grad = False

    def unfreeze_frozen_params(self):
        for local_name in self.frozen_local_ids:
            self.get_model().get_state_reference(local_name).requires_grad = True

    def is_model_loaded(self):
        return self.model_inst is not None

    def unload_model(self, save_model=True, unique_hash=False):
        if self.model_inst is None:
            return
        if save_model:
            self.save_model(unique_hash=unique_hash)

        if self.graph is not None:
            self.graph.entanglement_tracker.temporary_unregister_state(self)

        if self.model_inst.model:
            del self.model_inst.model
            self.model_inst.model = None

        del self.model_inst
        self.model_inst = None

        print("unloading {0}".format(self.output_dir))
        if self.lineage_train is not None:
            self.lineage_train.unload()
        
        clear_torch_cache()

    def create_global_id(self, param_name):
        return self.output_dir + "-" + param_name

    def get_local_ids_of_obj(self, obj_name):
        obj = self.get_model().get_state_reference(obj_name)
        if isinstance(obj, torch.nn.Module):
            local_ids = []
            for name in obj.state_dict():
                local_ids.append(obj_name + "." + name)
            return local_ids
        else:
            return [obj_name]

    def save_model(self, unique_hash=False):
        assert self.is_model_loaded()
        self.get_model().save(unique_hash=unique_hash)

    # ===============================================================================
    # Test methods: registration, running, get_test_results.
    # ===============================================================================

    def has_test(self, test):
        return test.name in self.test_name_list

    def remove_test(self, test):
        self.remove_test_by_name(test.name)

    def remove_test_by_name(self, test_name):
        del self.test_result_dict[test_name]
        self.test_name_list.remove(test_name)

    def add_test(self, test):
        assert isinstance(test, LineageTest)
        if not self.has_test(test):
            self.graph.test_dict[test.name] = test
            self.test_result_dict[test.name] = {
                "results": None,
                "success": None,
            }
            self.test_name_list.append(test.name)

    def run_test_by_name(self, test_name, return_results=False):
        assert test_name in self.test_name_list, "test not found!"
        return self.run_test(
            self.graph.test_dict[test_name], return_results=return_results
        )

    def run_test(self, test, return_results=False):
        assert isinstance(test, LineageTest)
        if not self.has_test(test):
            self.add_test(test)
        test_result, eval_results = test.run(self, return_results=True)
        self.test_result_dict[test.name] = {
            "results": eval_results,
            "success": test_result,
        }
        if return_results:
            return test_result, eval_results
        else:
            return test_result

    def is_test_failure(self):
        for test_name, result in self.test_result_dict.items():
            if result["success"] is None:
                self.run_test(self.graph.test_dict[test_name])
        return False in set(
            [result["success"] for result in self.test_result_dict.values()]
        )

    def run_all_tests(self, return_results=False):
        if self.model_inst is None:
            _ = self.get_model()
        self.model_inst.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        results = []
        for test_name in self.test_name_list:
            res = self.run_test_by_name(test_name, return_results=return_results)
            results.append(res)
        self.model_inst.model.to(device)
        clear_torch_cache()
        return results

    def get_test_result(self, test_name):
        assert isinstance(test_name, str)
        assert (
            test_name in self.graph.test_dict and test_name in self.test_name_list
        ), "test is not registered with {0}".format(self.output_dir)
        test = self.graph.test_dict[test_name]
        if self.test_result_dict[test_name]["results"] is None:
            self.run_test(self.graph.test_dict[test_name])
        eval_results = self.test_result_dict[test_name]["results"]
        if test.metric_for_best_model is not None:
            metric_for_best_model = test.metric_for_best_model
        else:
            metric_for_best_model = "loss"
        assert (
            metric_for_best_model in eval_results
            or "eval_" + metric_for_best_model in eval_results
        )
        if metric_for_best_model in eval_results:
            return eval_results[metric_for_best_model]
        else:
            return eval_results["eval_" + metric_for_best_model]

    # ===============================================================================
    # Helper functions for training.
    # ===============================================================================

    def get_trainer(self, re_initialize=False):
        if self.lineage_train is None:
            print("WARNING: no lineage train has been provided to this node")
            return None
        else:
            return self.lineage_train.get_trainer(re_initialize=re_initialize)

    def is_training_finished(self):
        if self.lineage_train is None:
            return True
        else:
            return self.lineage_train.is_finished()

    def train(self, re_initialize=False):
        if self.lineage_train is None:
            print("WARNING: no lineage train has been provided to this node")
            return
        elif not self.is_training_finished():
            assert isinstance(self.lineage_train, LineageTrain)
            self.is_dirty = True
            self.traced = None
            self.get_model().traced = None
            print(f"Training {self.output_dir}")
            self.lineage_train.run(re_initialize)
            print("WARNING: finished training don't forget to save!")
        else:
            print(
                "WARNING: called LineageNode.train() on node which already completed training"
            )

    # ===============================================================================
    # PyTorch model and tokenizer getters.
    # ===============================================================================

    def get_model(self, re_initialize=False, lineage_dataset=None):
        if self.model_inst is None or re_initialize is True:
            if lineage_dataset is None and self.lineage_train is not None:
                lineage_dataset = self.lineage_train.train_dataset
            self.initialize_model(lineage_dataset, re_initialize)
        return self.model_inst

    def get_pt_model(self):
        return self.get_model().model

    def get_pt_tokenizer(self):
        return self.get_model().tokenizer

    def initialize_model(self, lineage_dataset, re_initialize, device=device):
        """
        Create model instance represented by LineageModel and load the pytorch model via LineageModel.initialize().
        Infers task type according to lineage_dataset so users
        do not have to explicitly add head when adapting a model to a different task

        Parents of this node must complete training prior to attempt to initialize the model of this node

        if loading for first time or re_initialize is True, model will be re-initialized from either a init_checkpoint or parent node
        else if model is already stored then model is loaded from storage

        Args:
            lineage_dataset (LineageDataset or None): dataset associated with this node so the task type to load appropriate architecture type for task
            re_initialize (bool): if true, re-initializes the model associated with this node back to its initial state prior to training
        """
        for etype_iter in etypes:
            for parent in self.get_parents(etype_iter):
                if not parent.is_training_finished():
                    parent.train()
                assert (
                    parent.is_training_finished()
                ), "Cannot initialize model whose parents have not finished training!"

        if re_initialize is True and self.lineage_train is not None:
            self.lineage_train.reset()

        etype = "adapted"

        if isinstance(self.init_checkpoint, str):
            # Initialize model directly from checkpoint.
            self.model_inst = LineageModel(
                node=self,
                init_checkpoint=self.init_checkpoint,
                lineage_dataset=lineage_dataset,
                re_initialize=re_initialize,
            )
        elif isinstance(self.output_dir, str):
            # Initialization same as final model.
            print("WARNING: no initial checkpoint found for node")
            self.model_inst = LineageModel(
                node=self,
                init_checkpoint=None,
                lineage_dataset=lineage_dataset,
                re_initialize=re_initialize,
            )

        if (re_initialize or not self.is_init) and self.model_init_function is not None:
            print("WARNING: using user customized init_function")
            # Run custom user function.
            clear_torch_cache()
            self.model_init_function(self, self.get_parents(etype))

        assert (
            isinstance(self.model_inst, LineageModel)
            and self.model_inst.model is not None
        )

        self.model_inst.model.to(device)
        self.is_init = True

        if self.graph is not None:
            # assign gobal_ids to params for weight sharing
            self.graph.entanglement_tracker.enforce_entanglement(self)
            self.graph.entanglement_tracker.temporary_register_state(self)

        try:
            self.model_inst.model.config.model_type = self.model_type
        except AttributeError:
            pass

        self.traced = self.model_inst.traced

    def load_from_file(file_path):
        with open(file_path, "r") as f:
            config = json.load(f)
        return LineageNode.load_from_config(config)

    def load_from_config(config):
        live_attr = {"lineage_train"}
        if "lineage_train" in config:
            lineage_train = LineageTrain.load_from_config(config["lineage_train"])
        else:
            lineage_train = None

        persistent_args = dict((k, v) for k, v in config.items() if k not in live_attr)
        node = LineageNode(**persistent_args, lineage_train=lineage_train)
        return node

    def to_json(self):
        live_attr = {"lineage_train"}
        res = dict(
            (k, v)
            for k, v in self.__dict__.items()
            if misc_utils.is_jsonable({k: v}) and k != "traced"
        )
        res.update(
            (k, self.__dict__[k].to_json())
            for k in live_attr
            if self.__dict__[k] is not None
        )

        res["frozen_local_ids"] = list(self.frozen_local_ids)
        res.update((k, None) for k in live_attr if self.__dict__[k] is None)
        return res

    # ===============================================================================
    # Lineage graph traversal helper methods.
    # ===============================================================================

    def get_children(self, etype='adapted'):
        assert etype in etypes, "edge type must be one of {0}".format(etypes)
        return [
            self.graph.get_node(node_name) for node_name in self.children_dict[etype]
        ]

    def get_parents(self, etype='adapted'):
        assert etype in etypes, "edge type must be one of {0}".format(etypes)
        return [self.graph.get_node(node_name) for node_name in self.parent_dict[etype]]

    def get_latest_version(self):
        cur_node = self
        cur_children = cur_node.get_children("versioned")
        while cur_children:
            cur_node = cur_children[0]
            cur_children = cur_node.get_children("versioned")
        return cur_node

    # ===============================================================================
    # Lineage graph edge mutation methods.
    # ===============================================================================

    def remove_edge(self, target, etype):
        assert etype in etypes, "edge type must be one of {0}".format(etypes)
        assert (
            target.output_dir in self.parent_dict[etype]
            and self.output_dir in target.children_dict[etype]
        ), "({0},{1}) does not exist".format(target.output_dir, self.output_dir)

        self.parent_dict[etype].remove(target.output_dir)
        target.children_dict[etype].remove(self.output_dir)

        return True

    def add_edge(self, target, etype):
        assert etype in etypes, "edge type must be one of {0}".format(etypes)

        if etype == "versioned":
            assert (
                len(self.parent_dict["versioned"]) == 0
            ), "{0} is already versioned from {1}".format(
                self.output_dir, self.parent_dict["versioned"][0]
            )
            result, msg = self.is_versioned(target)
            if result:
                self.parent_dict["versioned"].append(target.output_dir)
                target.children_dict["versioned"].append(self.output_dir)
                return True, msg
            else:
                return False, msg

        elif etype == "adapted":
            assert (
                len(self.parent_dict["adapted"]) == 0
            ), "{0} is already adapted from {1}".format(
                self.output_dir, self.parent_dict["adapted"][0]
            )
            result, msg = self.is_adapted(target)
            if result:
                self.parent_dict["adapted"].append(target.output_dir)
                target.children_dict["adapted"].append(self.output_dir)
                return True, msg
            else:
                return False, msg

    def is_adapted(self, target):

        # check if there is already a parent
        if (
            len(self.parent_dict["adapted"]) > 0
            and target.output_dir not in self.parent_dict["adapted"]
        ):
            return (
                False,
                "{0} is already adapted from {1}".format(
                    self.output_dir, self.parent_dict["adapted"][0]
                ),
            )

        return True, ""

    def is_versioned(self, target):

        # check if there is already a parent
        if (
            len(self.parent_dict["versioned"]) > 0
            and target.output_dir not in self.parent_dict["versioned"]
        ):
            return (
                False,
                "{0} is already versioned from {1}".format(
                    self.output_dir, self.parent_dict["versioned"][0]
                ),
            )

        # self.get_model().model must be structurally the same as target.get_model().model
        add_nodes, del_nodes, add_edges, del_edges = diff_ht(
            [self.get_model().model, target.get_model().model]
        )

        if not (
            len(add_nodes) == len(del_nodes) == len(add_edges) == len(del_edges) == 0
        ):
            return (
                False,
                "{0} structurally differs from {1} on {2}".format(
                    self.output_dir,
                    target.output_dir,
                    add_nodes + del_nodes + add_edges + del_edges,
                ),
            )

        return True, ""


class LineageGraph:
    def __init__(self, compression_mode="lzma", single_model_compression=False, param_home="./parameter_store/"):
        self.root = []
        self.nodes = OrderedDict()
        self.log = defaultdict(dict)
        self.model_type_to_nodes = defaultdict(list)
        self.model_type_to_tests = defaultdict(list)
        self.test_dict = {}
        self.init_tracker = InitTracker()
        self.entanglement_tracker = EntanglementTracker()
        self.entanglement_tracker.graph = self
        self.single_model_compression = single_model_compression
        self.compression_mode = compression_mode
        self.param_home=param_home

    def get_node(self, node_name):
        return self.nodes[node_name]

    def get_all_nodes(self, parent='root'):
        if parent == 'root':
            all_nodes = []
            queue = deque(self.root)
            while len(queue):
                curr = queue.pop()
                all_nodes.append(self.get_node(curr))
                all_nodes += self.get_all_nodes(curr)
            return all_nodes
        else:
            if parent not in self.nodes:
                raise Exception(f'{parent} does not exist in the graph!')
            queue = deque([])
            for etype in etypes:
                for node in self.get_node(parent).get_children(etype=etype):
                    if node not in queue:
                        queue.append(node)
        all_nodes = []
        while len(queue):
            curr = queue.pop()
            all_nodes.append(curr)
            for etype in etypes:
                for node in curr.get_children(etype=etype):
                    if node not in queue:
                        queue.append(node)
        return all_nodes


    def get_pt_model(self, node_name):
        return self.nodes[node_name].get_pt_model()

    def register_test_to_graph(self, test):
        for node_name, _ in self.nodes.items():
            self.get_node(node_name).add_test(test)

    def register_test_to_node(self, test, node_name):
        self.get_node(node_name).add_test(test)

    def register_test_to_type(self, test, model_type):
        if test.name in self.test_dict:
            print("WARNING:", test.name, "already exists. test names must be unique")
        for node_name in self.model_type_to_nodes[model_type]:
            self.get_node(node_name).add_test(test)
        self.model_type_to_tests[model_type].append(test.name)
        self.test_dict[test.name] = test

    def remove_test(self, test):
        self.remove_test_by_name(test.name)

    def remove_test_by_name(self, test_name):
        for model_type in self.model_type_to_tests:
            for node_name in self.model_type_to_nodes[model_type]:
                self.get_node(node_name).remove_test_by_name(test_name)
            if test_name in self.model_type_to_tests[model_type]:
                self.model_type_to_tests[model_type].remove(test_name)
        del self.test_dict[test_name]

    def save(self, path, save_models=True, unique_hash=False):
        assert os.path.exists(path), path + " does not exist."
        if save_models:
            for node in self.nodes.values():
                if node.model_inst is not None:
                    node.save_model(unique_hash=unique_hash)
        config = {
            "root": self.root,
            "nodes": OrderedDict(
                {name: node.to_json() for name, node in self.nodes.items()}
            ),
            "log": {
                etype: {",".join(e): reason for e, reason in edges.items()}
                for etype, edges in self.log.items()
            },
            "entanglement_tracker": self.entanglement_tracker.to_json(),
            "init_tracker": self.init_tracker.to_json(),
            "model_type_to_nodes": self.model_type_to_nodes,
            "model_type_to_tests": self.model_type_to_tests,
            "test_dict": dict((k, v.to_json()) for k, v in self.test_dict.items()),
            "single_model_compression": self.single_model_compression,
            "compression_mode": self.compression_mode,
            "param_home" : self.param_home,
        }
        with open(os.path.join(path, "lineage_graph.json"), "w") as fp:
            json.dump(config, fp)

    def run_update_cascade(
        self, old_node, updated_node, skip_fn=None, terminate_fn=None
    ):
        assert updated_node.output_dir in [
            entry.output_dir for entry in old_node.get_children("versioned")
        ]
        node_to_visit = [
            node for node in old_node.get_children("adapted") if node != updated_node
        ]
        node_to_train = []
        while node_to_visit:
            cur_node = node_to_visit.pop(0)
            node_to_visit += cur_node.get_children("adapted")
            new_node = cur_node.copy(cur_node.output_dir + "_versioned")
            self.add(new_node, parent=cur_node.output_dir, etype="versioned")
            for cur_parent in cur_node.get_parents("adapted"):
                self.add(
                    new_node,
                    parent=cur_parent.get_latest_version().output_dir,
                    etype="adapted",
                )
            new_node.get_model()  # load in the model to register the global_ids of parameters
            new_node.unload_model()
            node_to_train.append(new_node)

        while node_to_train:
            cur_node = node_to_train.pop(0)
            if not cur_node.is_training_finished():
                cur_node.train()

    def update_cascade(
        self, old_base, new_base, old_target
    ):
        assert new_base.output_dir in [
            entry.output_dir for entry in old_base.get_children("versioned")
        ]
        subtree = self.get_all_nodes(parent=old_base.output_dir)
        assert old_target.output_dir in [
            entry.output_dir for entry in subtree
        ]
        node_to_visit = [
            node for node in old_base.get_children("adapted") if node != new_base
        ]
        node_to_train = []
        while node_to_visit:
            cur_node = node_to_visit.pop(0)
            node_to_visit += cur_node.get_children("adapted")
            new_node = cur_node.copy(cur_node.output_dir + "_versioned")
            self.add(new_node, parent=cur_node.output_dir, etype="versioned")
            for cur_parent in cur_node.get_parents("adapted"):
                self.add(
                    new_node,
                    parent=cur_parent.get_latest_version().output_dir,
                    etype="adapted",
                )
            new_node.get_model()  # load in the model to register the global_ids of parameters
            new_node.unload_model()
            node_to_train.append(new_node)

        while node_to_train:
            cur_node = node_to_train.pop(0)
            if not cur_node.is_training_finished():
                cur_node.train()

        new_target = self.get_node(old_target.output_dir + "_versioned")
        return new_target

    def load_from_file(path, load_tests=True, filename=""):
        if (filename == ""):
            file_path = os.path.join(path, "lineage_graph.json")
        else:
            file_path = os.path.join(path, filename)
        assert os.path.exists(file_path), file_path + " does not exist."
        with open(file_path, "r") as fp:
            config = json.load(fp)
        g = LineageGraph()
        g.load_from_config(config, load_tests=load_tests)
        return g

    def load_from_config(self, config, load_tests=True):
        self.entanglement_tracker = EntanglementTracker.load_from_config(
            config["entanglement_tracker"]
        )
        if "param_home" not in config:
            self.param_home = "./parameter_store/"
        self.entanglement_tracker.graph = self
        self.init_tracker = InitTracker.load_from_config(config["init_tracker"])
        self.root = config["root"]
        self.log = {
            etype: {
                (e.split(",")[0], e.split(",")[1]): reason
                for e, reason in edges.items()
            }
            for etype, edges in config["log"].items()
        }
        self.model_type_to_nodes = defaultdict(
            list, config["model_type_to_nodes"].items()
        )
        if load_tests:
            self.model_type_to_tests = defaultdict(
                list, config["model_type_to_tests"].items()
            )
            self.test_dict = dict(
                (k, LineageTest.load_from_config(v)) for k, v in config["test_dict"].items()
            )
        else:
            self.model_type_to_tests = {}
            self.test_dict = {}
        self.single_model_compression = config["single_model_compression"]
        self.compression_mode = config["compression_mode"]
        for name, node_config in config["nodes"].items():
            self.nodes[name] = LineageNode.load_from_config(node_config)
            self.temporary_register_node(self.nodes[name])

    def temporary_register_node(self, node):
        node.graph = self
        if (
            node.model_type is not None
            and node.model_type in self.model_type_to_nodes
            and node.output_dir not in self.model_type_to_nodes[node.model_type]
        ):
            self.model_type_to_nodes[node.model_type].append(node.output_dir)
        if node.model_type is not None and node.model_type in self.model_type_to_tests:
            for test_name in self.model_type_to_tests[node.model_type]:
                node.add_test(self.test_dict[test_name])

    def update(self, node):
        assert node in self.nodes, "{0} not found in the graph.".format(node.output_dir)
        msg_list = []
        parent_dict = defaultdict(list)
        children_dict = defaultdict(list)

        for etype, parent_list in node.parent_dict.items():
            parent = parent_list[0]
            if etype == "adapted":
                result, msg = node.is_adapted(self.nodes[parent])
            elif etype == "versioned":
                result, msg = node.is_verioned(self.nodes[parent])
            else:
                assert etype in etypes, "edge type must be one of {0}".format(etypes)

            if result:
                pass
            else:
                parent_dict[etype].append(parent)
                msg_list.append(
                    "previous {0} ({1},{2}) failed because: {3}".format(
                        etype, parent, node.output_dir, msg
                    )
                )

        for etype, children_list in node.children_dict.items():
            for child_name in children_list:
                child = self.nodes[child_name]
                if etype == "adapted":
                    result, msg = child.is_adapted(node)
                elif etype == "versioned":
                    result, msg = child.is_verioned(node)
                else:
                    assert etype in etypes, "edge type must be one of {0}".format(
                        etypes
                    )

                if result:
                    pass
                else:
                    children_dict[etype].append(child)
                    msg_list.append(
                        "previous {0} ({1},{2}) failed because: {3}".format(
                            etype, node.output_dir, child_name, msg
                        )
                    )

        if len(msg_list) == 0:
            return True

        print("update failed due to:\n")
        for msg in msg_list:
            print(msg)
        if len(parent_dict) != 0:
            print("suggesting detach from parents:")
            print(parent_dict)

        if len(children_dict) != 0:
            print("suggesting detach from children:")
            print(children_dict)

        return False

    def add_root(self, node):
        if len(self.root) != 0:
            for etype in etypes:
                if etype not in self.log:
                    self.log[etype] = {}
                self.log[etype][
                    ("root", node.output_dir)
                ] = "WARNING: {0} is specified to be the root without checking".format(
                    node.output_dir
                )
        else:
            for etype in etypes:
                if etype not in self.log:
                    self.log[etype] = {}
                self.log[etype][
                    ("root", node.output_dir)
                ] = "{0} becomes the root, as it is the first inserted node.".format(
                    node.output_dir
                )

        self.root.append(node.output_dir)
        self.nodes[node.output_dir] = node
        self.temporary_register_node(node)

    def append(
        self,
        node,
        etype=None,
        parent=None,
        input_names=None,
    ):
        assert isinstance(node, LineageNode), "can only insert LineageNode"
        if parent == "root":
            assert etype is None, "root insertion cannot specify edge type."
            self.add_root(node)
            print(self.log["adapted"][("root", node.output_dir)])
            return True

        elif len(self.root) == 0:
            assert parent is None, "parent {0} not found in the graph.".format(parent)
            assert etype is None, "root insertion cannot specify edge type."
            self.add_root(node)
            print(self.log["adapted"][("root", node.output_dir)])
            return True
        else:
            if parent is not None:
                assert (
                    parent in self.nodes
                ), "Parent {0} not found in the graph.".format(parent)
            if etype is None:
                etype = "adapted"
            assert etype in etypes, "edge type must be one of {0}".format(etypes)

        if etype == "versioned":

            if parent:
                assert len(node.parent_dict[etype]) == 0, (
                    node.output_dir + "has a parent already."
                )
                node.parent_dict[etype].append(parent)
                self.nodes[parent].children_dict[etype].append(node.output_dir)
                self.nodes[node.output_dir] = node
                self.log[etype][
                    (parent, node.output_dir)
                ] = "WARNING: {0} is specified to be versioned from {1} without checking".format(
                    node.output_dir, parent
                )
                print(self.log[etype][(parent, node.output_dir)])
                return True

        elif etype == "adapted":

            if parent:
                if len(node.parent_dict[etype]) > 0:
                    print(f"WARNING: {node.output_dir} has a parent already.")
                node.parent_dict[etype].append(parent)
                self.nodes[parent].children_dict[etype].append(node.output_dir)
                self.nodes[node.output_dir] = node
                self.log[etype][
                    (parent, node.output_dir)
                ] = "WARNING: {0} is specified to be adapted from {1} without checking".format(
                    node.output_dir, parent
                )
                print(self.log[etype][(parent, node.output_dir)])
                return True

            s_divergence_list = []
            c_divergence_list = []
            candidates = list(reversed(list(self.nodes.keys())))

            for name in candidates:
                c_divergence, s_divergence = divergence(
                    node, self.nodes[name], input_names
                )
                c_divergence_list.append(c_divergence)
                s_divergence_list.append(s_divergence)

            for name, i in sorted(
                zip(candidates, c_divergence_list),
                key=lambda t: (t[1], candidates.index(t[0])),
            ):
                if i == 0:#TODO: what if model A is the same as model B
                    continue
                if i >= 1:
                    print(
                        "there is no contextually similar node with {0}.".format(
                            node.output_dir
                        )
                    )
                    break

                result, msg = node.add_edge(self.nodes[name], etype)
                if result:
                    self.nodes[node.output_dir] = node
                    self.log[etype][
                        (name, node.output_dir)
                    ] = "{0} is adapted from {1} as they share weights".format(
                        node.output_dir, name
                    )
                    print(self.log[etype][(name, node.output_dir)])
                    return True
                else:
                    print(msg)

            for name, i in sorted(
                zip(candidates, s_divergence_list),
                key=lambda t: (t[1], candidates.index(t[0])),
            ):
                if i >= 1:
                    print(
                        "there is no structurally similar node with {0}.".format(
                            node.output_dir
                        )
                    )
                    break

                result, msg = node.add_edge(self.nodes[name], etype)
                if result:
                    self.nodes[node.output_dir] = node
                    self.log[etype][
                        (name, node.output_dir)
                    ] = "{0} is adapted from {1} as they share same layers".format(
                        node.output_dir, name
                    )
                    print(self.log[etype][(name, node.output_dir)])
                    return True
                else:
                    print(msg)

            print(
                "cannot insert {0} with edge type: {1} as there is no suitable parent.".format(
                    node.output_dir, etype
                )
            )
            return False

        if parent is None:

            for parent in self.root:

                result, msg = node.add_edge(self.nodes[parent], etype)
                if result:
                    self.nodes[node.output_dir] = node
                    if etype == "verioned":
                        self.log[etype][
                            (parent, node.output_dir)
                        ] = "{0} is verioned from {1} as they share the \
                            exact same structure".format(
                            node.output_dir, parent
                        )
                    else:
                        self.log[etype][
                            (parent, node.output_dir)
                        ] = "{0} shares same weight with {1} in {2}".format(
                            node.output_dir, parent, msg
                        )

                    print(self.log[etype][(parent, node.output_dir)])
                    return True
                else:
                    print(msg)

            print(
                "cannot insert {0} with edge type: {1} as there is no suitable parent.".format(
                    node.output_dir, etype
                )
            )
            return False

        else:
            result, msg = node.add_edge(self.nodes[parent], etype)
            if result:
                self.nodes[node.output_dir] = node
                if etype == "verioned":
                    self.log[etype][
                        (parent, node.output_dir)
                    ] = "{0} is verioned from {1} as they share the \
                        exact same structure".format(
                        node.output_dir, parent
                    )
                else:
                    self.log[etype][
                        (parent, node.output_dir)
                    ] = "{0} shares same weight with {1} in {2}".format(
                        node.output_dir, parent, msg
                    )

                print(self.log[etype][(parent, node.output_dir)])
                return True
            else:
                print(msg)

    def add(
        self,
        node,
        etype=None,
        parent=None,
        inplace=False,
        input_names=None,
    ):
        self.temporary_register_node(node)
        if inplace:
            assert parent is None, "Inplace update cannot specify parent node."
            assert etype is None, "Inplace update cannot specify edge type."
            success = self.update(node)
        else:
            success = self.append(node, etype, parent, input_names)
        return success

    def detach(self, output_dir):
        """
        Detaches node from current graph, assumes all nodes have only one parent
        """
        assert output_dir in self.nodes.keys(), "node {0} does not exist".format(
            output_dir
        )
        if output_dir in self.root:
            assert len(self.root) > 1, "Cannot detach the only root"
        node = self.nodes[output_dir]
        for etype in node.parent_dict.keys():
            for parent in node.parent_dict[etype]:
                unregister_init_from(self.nodes[parent], node)
                self.entanglement_tracker.remove_node(node)
                node.remove_edge(self.nodes[parent], etype)
                self.log[etype].pop((parent, output_dir))
        if node.is_delta:
            node.is_dirty = True
        del self.nodes[output_dir]
        return node

    def show(self, etype=None, save_path="./LineageGraph.html", layout=True):
        if not etype:
            etype = "adapted"
        assert etype in etypes, "edge type must be one of {0}".format(etypes)
        graph = {"root": self.root}
        for name in self.nodes.keys():
            graph[name] = self.nodes[name].children_dict[etype]

        graph = nx.DiGraph(graph)
        G = to_agraph(graph) 
        G.graph_attr["rankdir"] = "TB"
        G.graph_attr["splines"] = "ortho"
        G.graph_attr["ordering"] = "out"
        G.layout(prog='dot')
        G.draw(save_path[:-4] + 'pdf')

        net = Network(
            notebook=True,
            height="100%",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            layout=layout,
            directed=True,
        )
        # Set the hierarchical layout options
        net.set_options('''
        var options = {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed"
            }
          },
          "physics": {
            "enabled": false
          }
        }
        ''')
        net.from_nx(graph)
        for e in net.get_edges():
            if (e["from"], e["to"]) in self.log[etype]:
                e["title"] = self.log[etype][(e["from"], e["to"])]

        spl = dict(nx.all_pairs_shortest_path_length(graph))
        for n in net.get_nodes():
            net.get_node(n)["level"] = spl["root"][n]

        net.save_graph(save_path)
        return net


def ENTANGLE(src_node, dst_node, src_obj_name=None, dst_obj_name=None):
    if src_obj_name is None and dst_obj_name is None:
        for name in src_node.get_pt_model().state_dict():
            if name in dst_node.get_pt_model().state_dict():
                ENTANGLE(src_node, dst_node, src_obj_name=name)
        return
    elif dst_obj_name is None:
        dst_obj_name = src_obj_name
    obj = src_node.get_model().get_state_reference(src_obj_name)
    dst_node.get_model().set_state_reference(dst_obj_name, obj)
    assert len(src_node.local_to_global_id) == 0
    assert src_node.get_model().get_state_reference(
        src_obj_name
    ) is dst_node.get_model().get_state_reference(dst_obj_name)
    assert (
        dst_obj_name not in dst_node.local_to_global_id
        or src_obj_name not in src_node.local_to_global_id
    )
    src_local_names = src_node.get_local_ids_of_obj(src_obj_name)
    for name in src_local_names:
        if name in dst_node.local_to_global_id:
            src_node.local_to_global_id[name] = dst_node.local_to_global_id[name]
        elif src_obj_name in src_node.local_to_global_id:
            dst_node.local_to_global_id[name] = src_node.local_to_global_id[name]
        else:
            dst_node.local_to_global_id[name] = dst_node.create_global_id(name)
            src_node.local_to_global_id[name] = dst_node.local_to_global_id[name]


def INIT_FROM(src_node, dst_node, src_obj_name=None, dst_obj_name=None):
    if src_obj_name is None and dst_obj_name is None:
        for name in src_node.get_pt_model().state_dict():
            if name in dst_node.get_pt_model().state_dict():
                INIT_FROM(src_node, dst_node, src_obj_name=name)
        return
    elif dst_obj_name is None:
        dst_obj_name = src_obj_name
    obj = src_node.get_model().get_state_reference(src_obj_name)
    copy_obj = copy.deepcopy(obj)
    dst_node.get_model().set_state_reference(dst_obj_name, copy_obj)
    src_local_names = src_node.get_local_ids_of_obj(src_obj_name)
    # assume src_name == dst_name
    graph = src_node.graph
    for name in src_local_names:
        dst_node.local_to_global_id[name] = dst_node.create_global_id(name)
        graph.init_tracker.set_init_from(
            dst_node.local_to_global_id[name], src_node.local_to_global_id[name]
        )


def register_init_from(
    src_node, dst_node, src_obj_name=None, dst_obj_name=None, name_map=None
):
    if name_map:
        convert = lambda x: name_map[x]
    else:
        convert = lambda x: x

    if src_obj_name is None and dst_obj_name is None:

        def equal(x, y):
            return x[1] == y[1]

        src_layers = list(src_node.get_pt_model().state_dict().keys())
        dst_layers = list(dst_node.get_pt_model().state_dict().keys())

        a = [
            (name, src_node.get_pt_model().state_dict()[name].shape)
            for name in src_layers
        ]
        b = [
            (name, dst_node.get_pt_model().state_dict()[name].shape)
            for name in dst_layers
        ]

        name_map = {src_layers[p[0]]: dst_layers[p[1]] for p in lcs_one(a, b, equal)}
        for name in name_map.keys():
            register_init_from(
                src_node,
                dst_node,
                src_obj_name=name,
                dst_obj_name=name_map[name],
                name_map=name_map,
            )
        return

    else:
        src_local_names = src_node.get_local_ids_of_obj(src_obj_name)
        graph = src_node.graph
        for name in src_local_names:
            dst_node.local_to_global_id[convert(name)] = dst_node.create_global_id(
                convert(name)
            )
            graph.init_tracker.set_init_from(
                dst_node.local_to_global_id[convert(name)],
                src_node.local_to_global_id[name],
            )


def unregister_init_from(
    src_node, dst_node, src_obj_name=None, dst_obj_name=None, name_map=None
):
    if name_map:
        convert = lambda x: name_map[x]
    else:
        convert = lambda x: x

    if src_obj_name is None and dst_obj_name is None:

        def equal(x, y):
            return x[1] == y[1]

        src_layers = list(src_node.get_pt_model().state_dict().keys())
        dst_layers = list(dst_node.get_pt_model().state_dict().keys())

        a = [
            (name, src_node.get_pt_model().state_dict()[name].shape)
            for name in src_layers
        ]
        b = [
            (name, dst_node.get_pt_model().state_dict()[name].shape)
            for name in dst_layers
        ]

        name_map = {src_layers[p[0]]: dst_layers[p[1]] for p in lcs_one(a, b, equal)}
        for name in name_map.keys():
            unregister_init_from(
                src_node,
                dst_node,
                src_obj_name=name,
                dst_obj_name=name_map[name],
                name_map=name_map,
            )
        return

    else:
        src_local_names = src_node.get_local_ids_of_obj(src_obj_name)
        graph = src_node.graph
        for name in src_local_names:
            graph.init_tracker.unset_init_from(
                dst_node.local_to_global_id[convert(name)],
                src_node.local_to_global_id[name],
            )


def FREEZE(node, obj_name):
    local_names = node.get_local_ids_of_obj(obj_name)
    node.frozen_local_ids.update(local_names)


class InitTracker:
    """
    Tracks the initialization relationship between parameters as parameters can be initialized from another
    """

    def __init__(self):
        self.global_init_from = {}  # map from global_id to initial_parameter_global_id
        self.global_init_into = defaultdict(
            set
        )  # map from global_id to set of parameters initialized from global_id

    def get_init_from(self, global_id):
        assert global_id in self.global_init_from
        return self.global_init_from[global_id]

    def get_init_into(self, global_id):
        assert global_id in self.global_init_into
        return self.global_init_into[global_id]

    def set_init_from(self, dst_global_id, src_global_id):
        """
        register that a parameter is initialized from another.
        dst_global_id is initialized from src_global_id

        Args:
            dst_global_id: global_id of parameter initialized from src_global_id
            src_global_id: global_id of parameter which represents the initial value of dst_global_id
        """
        self.global_init_from[dst_global_id] = src_global_id
        self.global_init_into[src_global_id].add(dst_global_id)

    def unset_init_from(self, dst_global_id, src_global_id):
        if (
            dst_global_id in self.global_init_from
            and src_global_id in self.global_init_into
            and dst_global_id in self.global_init_into[src_global_id]
        ):
            del self.global_init_from[dst_global_id]
            self.global_init_into[src_global_id].remove(dst_global_id)

    def to_json(self):
        attr_to_save = {"global_init_from", "global_init_into"}
        res = dict(
            (k, json.loads(json.dumps(self.__dict__[k], default=set_default)))
            for k in attr_to_save
        )
        return res

    def load_from_config(config):
        attr_to_load = {"global_init_from", "global_init_into"}
        init_tracker = InitTracker()
        init_tracker.__dict__.update(
            (k, v) for k, v in config.items() if k in attr_to_load
        )
        init_tracker.global_init_into = defaultdict(
            set, [(k, set(v)) for k, v in init_tracker.global_init_into.items()]
        )
        return init_tracker


class EntanglementTracker:
    """
    Tracks weight sharing across nodes where updates to shared weights must be synchronized across relevant nodes
    """

    def __init__(self):
        self.entangle_map = {}
        self.global_to_hash = {}
        self.hash_to_global = defaultdict(set)

        self.graph = None

        self.temporary_parameters = {}
        self.temporary_parameter_ref_count = defaultdict(set)

    def has_global_ids(self, global_ids):
        """
        Indicates whether the entanglement_tracker contains a set of global_ids

        Args:
            global_ids: set/list of global_ids

        Returns:
            bool
        """
        for global_id in global_ids:
            if global_id not in self.global_to_hash:
                return False
        return True

    def load_global_dict(self, global_ids):
        """
        loads a dictionary of global_id to parameter with weight sharing and delta compression optimizations

        Args:
            global_ids: set/list of global_ids to load

        Returns:
            dictionary of items (global_id, tensor of parameter)
        """
        global_dict = {}

        def helper(global_id):
            if global_id in self.temporary_parameters:
                return self.temporary_parameters[global_id].detach()

            file_path = os.path.join(self.graph.param_home, self.global_to_hash[global_id])
            tmp_hash_dict = torch.load(file_path, map_location = torch.device('cpu'))

            if not tmp_hash_dict["is_delta"]:
                return tmp_hash_dict["param"]
            else:
                if not self.graph.single_model_compression:
                    parent_param_id = self.graph.init_tracker.global_init_from[
                        global_id
                    ]
                    parent_param = helper(parent_param_id)
                    device = parent_param.device

                compressed = tmp_hash_dict["param"]
                quantize_delta = tmp_hash_dict["quantize_delta"]

                if self.graph.compression_mode == "rle":
                    values = compressed["values"]
                    counts = compressed["counts"]
                    shape = compressed["shape"]
                    delta = torch.tensor(
                        decompress_rle(
                            values, counts, shape, quantize_delta=quantize_delta
                        )
                    ).to(device)
                elif self.graph.compression_mode == "lzma":
                    x = compressed["x"]
                    shape = compressed["shape"]
                    delta = torch.tensor(
                        decompress_lzma(x, shape, quantize_delta=quantize_delta)
                    ).to(device)
                elif self.graph.compression_mode == "sparse":
                    x = compressed["x"]
                    delta = decompress_sparse(x, quantize_delta=quantize_delta).to(
                        device
                    )
                elif self.graph.compression_mode == "dict":
                    coded = compressed["coded"]
                    r_code_book = compressed["r_code_book"]
                    delta = decompress_dict(
                        coded, r_code_book, quantize_delta=quantize_delta
                    ).to(device)
                elif self.graph.compression_mode == "sparse_dict":
                    indices = compressed["indices"]
                    values = compressed["values"]
                    size = compressed["size"]
                    delta = decompress_sparse_dict(
                        indices, values, size, quantize_delta=quantize_delta
                    ).to(device)
                else:
                    t_1 = compressed["t_1"]
                    t_2 = compressed["t_2"]
                    shape = compressed["shape"]
                    delta = torch.tensor(
                        decompress(t_1, t_2, shape, quantize_delta=quantize_delta)
                    ).to(device)

                if self.graph.single_model_compression:
                    if quantize_delta:
                        return delta
                    else:
                        return dequantize(delta)
                else:
                    if quantize_delta:
                        return delta + parent_param
                    else:
                        return dequantize(delta + quantize(parent_param))

        for global_id in global_ids:
            global_dict.update({global_id: helper(global_id)})
        return global_dict

    # TODO hash collision
    def hash_param(self, param, node_name=None):
        """
        computes a content-based hash of the input tensor param

        Args:
            param: tensor

        Returns:
            string: the content hash of param
        """
        np_param = copy.deepcopy(param).cpu().numpy()
        res = (
            ('{0}_'.format(node_name) if node_name else '')
            + sha256(np_param.tobytes()).hexdigest()
            + "_"
            + ",".join([str(entry) for entry in np_param.shape])
        )
        file_path = os.path.join(self.graph.param_home, res)
        if os.path.exists(file_path):
            existing_param_dict = torch.load(file_path, map_location=torch.device('cpu'))
            if not existing_param_dict["is_delta"] and not torch.equal(
                param.cpu(), existing_param_dict["param"]
            ):
                assert (
                    False
                ), "parameter hash collision for parameters that are not equal"
        return res

    def remove_node(self, node):
        for global_id in node.local_to_global_id.values():
            if node.output_dir in self.entangle_map:
                self.entangle_map[global_id].remove(node.output_dir)
            if len(self.entangle_map[global_id]) == 0:
                self.remove_param_store(global_id)

    def update_param_store(self, global_id, param, node_name=None):
        """
        Computes a content-based hash of the input tensor param, stores the tensor to disk if
        not yet on disk. The previous hash of global_id is checked and removed if no other
        parameters use it

        Args:
            global_id: global_id of param
            param: tensor to store
        """
        cur_hash = self.hash_param(param, node_name)
        if global_id in self.global_to_hash:
            prev_hash = self.global_to_hash[global_id]
            if cur_hash != prev_hash:
                if len(self.hash_to_global[prev_hash]) == 1:
                    self.delete_param_store(prev_hash)
                self.hash_to_global[prev_hash].remove(global_id)
        if len(self.hash_to_global[cur_hash]) == 0:
            self.save_param_store(cur_hash, param)
        self.global_to_hash[global_id] = cur_hash
        self.hash_to_global[cur_hash].add(global_id)

    def update_delta_store(
        self, global_id, compressed, decompressed, quantize_delta=True, node_name=None
    ):
        """
        Computes a content-based hash of a delta from delta compression and stores the tensor to disk if
        not yet on disk. The previous hash of global_id is checked and removed if no other parameters use it

        Args:
            global_id: global_id of param
            compressed: tensor of compressed delta (NOT actually stored to disk)
            decompressed: tensor of decompressed delta to store
        """

        cur_hash = self.hash_param(decompressed, node_name)
        if global_id in self.global_to_hash:
            prev_hash = self.global_to_hash[global_id]
            if cur_hash != prev_hash:
                if len(self.hash_to_global[prev_hash]) == 1:
                    self.delete_param_store(prev_hash)
                self.hash_to_global[prev_hash].remove(global_id)
        if len(self.hash_to_global[cur_hash]) == 0:
            self.save_delta_store(cur_hash, compressed, quantize_delta)
        self.global_to_hash[global_id] = cur_hash
        self.hash_to_global[cur_hash].add(global_id)

    def remove_param_store(self, global_id):
        if global_id in self.global_to_hash:
            cur_hash = self.global_to_hash[global_id]
            if len(self.hash_to_global[cur_hash]) == 1:
                self.delete_param_store(cur_hash)
            self.hash_to_global[cur_hash].remove(global_id)
            del self.global_to_hash[global_id]

    def delete_param_store(self, param_hash):
        """
        delete saved parameter from file system. the filename of the parameter is the content-based
        hash of the parameter

        Args:
            param_hash: hash of parameter to delete, hash is computed by hash_param()
        """

        file_path = os.path.join(self.graph.param_home, param_hash)
        os.remove(file_path)

    def save_param_store(self, param_hash, param):
        """
        save tensor to file system, the filename is param_hash, which is the content-based
        hash of param

        Args:
            param_hash: content hash of param
            param: tensor to store
        """

        file_path = os.path.join(self.graph.param_home, param_hash)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({"param": param, "is_delta": False}, file_path)

    def save_delta_store(self, param_hash, compressed, quantize_delta=True):
        """
        save delta to file system, the filename is param_hash, which is the content-based
        hash of decompressed

        Args:
            param_hash: content-based hash of decompressed
            compressed: delta to store
        """
        
        file_path = os.path.join(self.graph.param_home, param_hash)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if self.graph.compression_mode == "rle":
            values, counts, shape = compressed
            torch.save(
                {
                    "param": {"values": values, "counts": counts, "shape": shape},
                    "is_delta": True,
                    "quantize_delta": quantize_delta,
                },
                file_path,
            )
        elif self.graph.compression_mode == "lzma":
            x, shape = compressed
            torch.save(
                {
                    "param": {"x": x, "shape": shape},
                    "is_delta": True,
                    "quantize_delta": quantize_delta,
                },
                file_path,
            )
        elif self.graph.compression_mode == "sparse":
            x = compressed
            torch.save(
                {
                    "param": {"x": x},
                    "is_delta": True,
                    "quantize_delta": quantize_delta,
                },
                file_path,
            )
        elif self.graph.compression_mode == "dict":
            coded, r_code_book = compressed
            torch.save(
                {
                    "param": {"coded": coded, "r_code_book": r_code_book},
                    "is_delta": True,
                    "quantize_delta": quantize_delta,
                },
                file_path,
            )
        elif self.graph.compression_mode == "sparse_dict":
            indices, values, size = compressed
            torch.save(
                {
                    "param": {"indices": indices, "values": values, "size": size},
                    "is_delta": True,
                    "quantize_delta": quantize_delta,
                },
                file_path,
            )
        else:
            t_1, t_2, shape = compressed
            torch.save(
                {
                    "param": {"t_1": t_1, "t_2": t_2, "shape": shape},
                    "is_delta": True,
                    "quantize_delta": quantize_delta,
                },
                file_path,
            )

    def to_json(self):
        attr_to_save = {"entangle_map", "global_to_hash", "hash_to_global"}
        res = dict(
            (k, json.loads(json.dumps(self.__dict__[k], default=set_default)))
            for k in attr_to_save
        )
        return res

    def load_from_config(config):
        attr_to_load = {"entangle_map", "global_to_hash", "hash_to_global"}
        entanglement_tracker = EntanglementTracker()
        entanglement_tracker.__dict__.update(
            (k, v) for k, v in config.items() if k in attr_to_load
        )
        entanglement_tracker.hash_to_global = defaultdict(
            set,
            [(k, set(v)) for k, v in entanglement_tracker.hash_to_global.items()],
        )
        return entanglement_tracker

    def temporary_register_state(self, node):
        for name in node.get_model().model.state_dict():
            param = node.get_model().get_state_reference(name)
            self.inner_temporary_register_state(node, name, param)

    def inner_temporary_register_state(self, node, name, param):
        if name not in node.local_to_global_id:
            node.local_to_global_id[name] = node.create_global_id(name)

        global_id = node.local_to_global_id[name]
        if global_id not in self.entangle_map:
            self.entangle_map[global_id] = [node.output_dir]
        elif (
            global_id in self.entangle_map
            and node.output_dir not in self.entangle_map[global_id]
        ):
            self.entangle_map[global_id].append(node.output_dir)

        self.temporary_parameters[global_id] = param
        self.temporary_parameter_ref_count[global_id].add(node.get_model())

    def enforce_entanglement(self, node):
        if len(node.local_to_global_id) > 0:
            assert node.output_dir in self.graph.nodes
            for name in node.get_model().model.state_dict():
                param = node.get_model().get_state_reference(name)
                self.inner_enforce_entanglement(node, name, param)

    def inner_enforce_entanglement(self, node, name, param):
        if name in node.local_to_global_id:
            global_id = node.local_to_global_id[name]
            cur_model = node.get_model().model
            if global_id in self.temporary_parameters:
                node.get_model().set_state_reference(
                    name, self.temporary_parameters[global_id]
                )
            else:
                self.temporary_parameters[global_id] = param
                self.temporary_parameter_ref_count[global_id].add(node.get_model())

    def temporary_unregister_state(self, node):
        tmp_ids_to_remove = set()
        for local_id, global_id in node.local_to_global_id.items():
            param_owner_set = self.temporary_parameter_ref_count[global_id]
            if node.get_model() in param_owner_set:
                param_owner_set.remove(node.get_model())
            if len(param_owner_set) == 0:
                param = node.get_model().get_state_reference(local_id)
                if global_id in self.temporary_parameters:
                    self.temporary_parameters.pop(global_id)

    def get_entangled_step(self, node):
        res = set()
        for global_param_id in node.local_to_global_id.values():
            res = res.union(set(self.entangle_map[global_param_id]))
        return set([self.graph.get_node(entry) for entry in res])

    def get_entangled(self, node):
        """
        get the set of all nodes where each node shares atleast one parameter with another in the set
        """
        if len(node.local_to_global_id) == 0:
            if not node.is_model_loaded():
                node.get_model()
        res = self.get_entangled_step(node)
        cur_new_nodes = res
        while len(cur_new_nodes) > 0:
            prev_new_nodes = cur_new_nodes.copy()
            cur_new_nodes = set()
            for node in prev_new_nodes:
                entangled_nodes = self.get_entangled_step(node)
                cur_new_nodes = cur_new_nodes.union(entangled_nodes.difference(res))
                res = res.union(entangled_nodes)
        assert (
            len(res) > 0
        ), "something went wrong: a node is atleast entangled with itself"
        return res


def lineage_node_assertion(graph, node):
    """
    Assert that node is in graph.
    """
    if isinstance(node, LineageNode):
        assert node.graph is not None
        assert graph is node.graph
        assert node.output_dir in graph.nodes
        assert node is graph.nodes[node.output_dir]
    elif isinstance(node, str):
        assert node in graph.nodes
    else:
        assert False, "node is not valid"


def get_all_ancestors(start_node, etype="adapted"):
    """
    Get all ancestors of start_node while only following edges of type etype.
    """
    res = set()
    if len(start_node.parent_dict[etype]) > 0:
        next_nodes = set(start_node.get_parents(etype))
        res = res.union(next_nodes)
        for node in next_nodes:
            res = res.union(get_all_ancestors(node))
    return res


def get_all_offspring(start_node, etype="adapted"):
    """
    Get all offspring of start_node while only following edges of type etype.
    """
    res = set()
    if len(start_node.children_dict[etype]) > 0:
        next_nodes = set(start_node.get_children(etype))
        res = res.union(next_nodes)
        for node in next_nodes:
            res = res.union(get_all_offspring(node))
    return res

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/workspace/HF_cache/'
os.environ['HF_DATASETS_CACHE'] = '/workspace/HF_cache/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/HF_cache/transformers_cache/'
import torch
from torch.nn import functional as F
from torch import Tensor
import sys
import time
import math

MGIT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(MGIT_PATH)
from typing import Any, Callable, Dict, Tuple, Optional
from utils.lineage.graph import *
from utils.ht.diffcheck import diff_ht_helper
from utils.ht.difflib import *
from utils.model_utils import smart_embedding_resize
from lcs.diffcheck import lcs_one
from utils import meta_functions
import transformers
from transformers import AutoModelForCausalLM, PreTrainedModel
# To control logging level for various modules used in the application:
import logging
import re
import gc
import argparse
import shutil
from lm_eval.tasks.hendrycks_test import SUBJECTS
from lm_eval import evaluator
import collections
import GPUtil
import multiprocess as mp
import json
from tqdm import tqdm


def aggregate_hendrycksTest(result_dict):
    count_dict = collections.defaultdict(int)
    for k in list(result_dict['results'].keys()):
        if "hendrycksTest" in k:
            name = f"hendrycksTest_{k.split('_')[-1]}"
            count_dict[name] += 1
            if name not in result_dict['results']:
                result_dict['results'][name] = {'acc': result_dict['results'][k]["acc"]}
            else:
                result_dict['results'][name]['acc'] += result_dict['results'][k]["acc"]
            del result_dict['results'][k]
    for k, v in count_dict.items():
        result_dict['results'][k]['acc'] = result_dict['results'][k]['acc'] / v
    return result_dict


def aggregate_truthfulqa_mc(result_dict):
    for k in list(result_dict['results'].keys()):
        if "truthfulqa_mc" in k:
            result_dict['results'][k] = {'mc_2': result_dict['results'][k]['mc2']}
    return result_dict


def aggregate_hellaswag(result_dict):
    for k in list(result_dict['results'].keys()):
        if "hellaswag" in k:
            result_dict['results'][k] = {'acc_norm': result_dict['results'][k]['acc_norm']}
    return result_dict


def aggregate_arc_challenge(result_dict):
    for k in list(result_dict['results'].keys()):
        if "arc_challenge" in k:
            result_dict['results'][k] = {'acc_norm': result_dict['results'][k]['acc_norm']}
    return result_dict


def aggregate_index(result_dict, index):
    for k in list(result_dict['results'].keys()):
        if k[-2:] != f"_{index}":
            del result_dict['results'][k]
    return result_dict


post_process_result = [aggregate_hendrycksTest, aggregate_truthfulqa_mc, aggregate_hellaswag, aggregate_arc_challenge]


def run_test(model, tokenizer, tasks, batch_size, test_limit, post_process_result=post_process_result):
    print("using gpu: ", [i for i in range(torch.cuda.device_count())])
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    model.device = model.module.device
    model.config = model.module.config
    result = {}
    result["results"] = collections.defaultdict(dict)
    result["versions"] = collections.defaultdict(dict)
    num_fewshots = {"arc_challenge": 25,
                    "hellaswag": 10,
                    "truthfulqa_mc": 0,
                    "hendrycksTest": 5,
                    }
    start = time.time()
    for t in tasks:
        torch.cuda.empty_cache()
        t_start = time.time()
        num_fewshot = 0
        for k_, v_ in num_fewshots.items():
            if k_ in t:
                num_fewshot = v_
        results = evaluator.simple_evaluate(
            model=model,
            tokenizer=tokenizer,
            tasks=[t],
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            limit=test_limit,
        )

        for k, v in results["results"].items():
            result["results"][k] = v
        for k, v in results["versions"].items():
            result["versions"][k] = v
        if "config" not in result:
            result["config"] = results["config"]
            result["config"]["num_fewshot"] = [num_fewshot]
        else:
            result["config"]["num_fewshot"].append(num_fewshot)
        clear_torch_cache()
        print(f"{t} finishes in {time.time() - t_start}")

    model = model.module
    clear_torch_cache()
    for post_process in post_process_result:
        result = post_process(result)
    print(f"test finishes in {time.time() - start}")
    return result


def run_collocated_workload(model, seq_len, b_sizes, max_new_tokens, nb_iters=10, warmup_iters=4):
    print_memory_allocated()
    return
    print("Running Latency Test:")
    for batch_size in tqdm(b_sizes):
        t_cost = []

        for i in range(nb_iters):
            tensor = torch.ones(batch_size * 2, seq_len, dtype=torch.int64).to("cuda")
            torch.cuda.synchronize()
            # # start profiling after 10 warmup iterations
            # if i >= warmup_iters: torch.cuda.cudart().sequential()
            # # push range for current iteration
            # if i >= warmup_iters: torch.cuda.nvtx.range_push("iteration{} for batch size {}".format(i, batch_size))

            if i >= warmup_iters: s = time.time()

            _ = model.generate(tensor, max_new_tokens=max_new_tokens)

            torch.cuda.synchronize()
            if i >= warmup_iters: t_cost.append(time.time() - s)
            # pop iteration range
            # if i >= warmup_iters: torch.cuda.nvtx.range_pop()

        print(
            f"batch_size = {batch_size * 2} | sequence_length = {seq_len} | single_batch_latency = {sum(t_cost) / len(t_cost)}\n")

    # torch.cuda.cudart().cudaProfilerStop()
    return


def run_sequential_workload(models, seq_len, b_sizes, max_new_tokens, nb_iters=10, warmup_iters=4):
    print_memory_allocated()
    for batch_size in tqdm(b_sizes):
        t_cost = []

        for i in range(nb_iters):
            models[1].to('cpu')
            models[0].to('cuda')

            tensors = [torch.ones(batch_size, seq_len, dtype=torch.int64).to("cuda") for _ in range(2)]
            # tensors = [torch.ones(batch_size*2, seq_len, dtype=torch.int64).to("cuda") for _ in range(2)]
            torch.cuda.synchronize()
            # # start profiling after 10 warmup iterations
            # if i >= warmup_iters: torch.cuda.cudart().cudaProfilerStart()
            # # push range for current iteration
            # if i >= warmup_iters: torch.cuda.nvtx.range_push("iteration{} for batch size {}".format(i, batch_size))

            if i >= warmup_iters: s = time.time()

            _ = models[0].generate(tensors[0], max_new_tokens=max_new_tokens)
            # _ = models[0].generate(tensors[1], max_new_tokens=max_new_tokens)
            models[0].to('cpu')
            models[1].to('cuda')
            torch.cuda.synchronize()
            _ = models[1].generate(tensors[1], max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            if i >= warmup_iters: t_cost.append(time.time() - s)
            # pop iteration range
            # if i >= warmup_iters: torch.cuda.nvtx.range_pop()

        print(
            f"batch_size = {batch_size * 2} | sequence_length = {seq_len} | single_batch_latency = {sum(t_cost) / len(t_cost)}\n")

    # torch.cuda.cudart().cudaProfilerStop()
    return


def run_concurrent_workload(models, seq_len, b_sizes, max_new_tokens, nb_iters=400, warmup_iters=100):
    barrier = mp.Barrier(2)
    manager = mp.Manager()
    result = manager.dict()

    def run(model, index, result, m):
        result[index] = m.dict()
        model = model.to("cuda")
        t_cost = {}
        for batch_size in b_sizes:
            t_cost[batch_size] = []
            torch.cuda.synchronize()
            barrier.wait()
            for i in range(nb_iters):
                if i == warmup_iters: print(f"{index} | start: {time.time()}\n")
                if i >= warmup_iters: s = time.time()
                _ = model.generate(torch.ones(batch_size, seq_len, dtype=torch.int64).to("cuda"),
                                   max_new_tokens=max_new_tokens)
                if i >= warmup_iters: t_cost[batch_size].append(time.time() - s)
            torch.cuda.synchronize()
            print(f"{index} | end: {time.time()}\n")
            barrier.wait()
            result[index][batch_size] = t_cost[batch_size]
            print(f"{index} | batch_size = {batch_size} \
                  | sequence_length = {seq_len} \
                  | single_batch_latency = {sum(t_cost[batch_size]) / len(t_cost[batch_size])}\n")

    p1 = mp.Process(target=run, args=(models[0], 1, result, manager))
    p2 = mp.Process(target=run, args=(models[1], 2, result, manager))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    valid_len = nb_iters - 2 * warmup_iters
    for b_size in tqdm(b_sizes):
        r1 = result[1][b_size]
        r2 = result[2][b_size]
        print(f"  | batch_size = {b_size * 2} \
                  | sequence_length = {seq_len} \
                  | single_batch_latency = {(sum(r1[:valid_len]) + sum(r2[:valid_len])) / (2 * valid_len)}\n")
    return


class CollocatedModel(PreTrainedModel):
    def __init__(self, model, dup_factor=1, print_cuda_after_forward=False):
        super(CollocatedModel, self).__init__(model.config)
        self.config._name_or_path = f'collocated_{model.config._name_or_path}'
        self.model = model
        self.inference_time = []
        self.dup_factor = dup_factor

        if self.dup_factor == 1:
            self.process = self.lambda_func
        else:
            self.process = self.duplicate

        if print_cuda_after_forward:
            def new_forward(self, *args, **kwargs):
                out = self.old_forward(*args, **kwargs)
                print_memory_allocated()
                return out

            bound_method = new_forward.__get__(self.model, self.model.__class__)
            setattr(self.model, 'old_forward', self.model.forward)
            setattr(self.model, 'forward', bound_method)

        self.model.eval()

    def lambda_func(self, *args):
        return args

    def duplicate(self, *args):
        args = list(args)
        args[0] = args[0].repeat(self.dup_factor, 1)
        return tuple(args)

    # def forward(self, *args, **kwargs):
    #     args = self.process(*args)
    #     torch.cuda.synchronize()
    #     start = time.time()
    #     out = self.model.forward(*args, **kwargs)
    #     torch.cuda.synchronize()
    #     self.inference_time.append(time.time()- start)
    #     return out

    # def generate(self, *args, **kwargs):
    #     print('call generate()')
    #     args = self.process(*args)
    #     torch.cuda.synchronize()
    #     start = time.time()
    #     out = self.model.generate(*args, **kwargs)
    #     torch.cuda.synchronize()
    #     self.inference_time.append(time.time()- start)
    #     return out

    def forward(self, *args, **kwargs):
        args = self.process(*args)
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        args = self.process(*args)
        return self.model.generate(*args, **kwargs)

    def get_total_runtime(self):
        return sum(self.inference_time)

    def get_total_num_inference(self):
        return len(self.inference_time)

    def clear_stats(self):
        self.inference_time = []


def print_memory_allocated():
    print(f"cuda memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
    print(f"cuda memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024} MB")


parser = argparse.ArgumentParser(description="Model Collocation")
parser.add_argument('--models',
                    type=str,
                    nargs='+',
                    help='model names'
                    )
parser.add_argument(
    "--mode",
    default="collocate",
    choices=["collocate", "sequence", "concurrent"],
    type=str,
)
parser.add_argument(
    "--auth_token",
    type=str,
)
parser.add_argument(
    "--delta_path",
    default='./param_change_07.bin',
    type=str,
)
parser.add_argument(
    "--collocate_percentage",
    default=0.,
    type=float,
)
parser.add_argument(
    "--diff_threshold",
    default=0.05,
    type=float,
)
parser.add_argument(
    "--prune_threshold",
    default=0,
    type=float,
)
parser.add_argument(
    "--profile_mode",
    default="performance",
    choices=["latency", "performance"],
    type=str,
)
parser.add_argument('--seq_len',
                    type=int,
                    default=128,
                    help='sequence length for latency evaluation'
                    )
parser.add_argument('--b_sizes',
                    nargs='+',
                    default=[1, 2, 4, 8, 16, 32, 64],
                    help='list of batch sizes for latency evaluation'
                    )
parser.add_argument('--max_new_tokens',
                    type=int,
                    default=1024,
                    help='max number of new tokens to generate for latency evaluation'
                    )
parser.add_argument('--batch_size',
                    type=int,
                    default=3,
                    help='batch size for performance evaluation'
                    )
parser.add_argument('--test_limit',
                    default=None,
                    help='test portion for performance evaluation'
                    )
parser.add_argument('--tasks',
                    type=str,
                    nargs='+',
                    default=['hendrycksTest'],
                    # default=['hendrycksTest', 'arc_challenge','hellaswag', 'truthfulqa_mc','pile_arxiv'],
                    help='tasks for performance evaluation'
                    )


def main(args):
    batch_size = args.batch_size if args.batch_size != 0 else "auto"
    tasks = args.tasks

    if 'hendrycksTest' in tasks:
        tasks.remove('hendrycksTest')
        tasks += [f"hendrycksTest-{sub}" for sub in SUBJECTS]
    models = args.models

    temp = time.time()
    model = AutoModelForCausalLM.from_pretrained(models[0], torch_dtype=torch.float16,
                                                 trust_remote_code=True, low_cpu_mem_usage=True,
                                                 use_auth_token=args.auth_token)
    print(f"loading {models[0]} to cpu costs {time.time() - temp}\n")

    if len(models) == 2 and args.mode == "collocate":
        if args.collocate_percentage != 0.:
            args.delta_path = f"./diff/{models[0].split('/')[-1] if models[0].split('/')[-1] != '' else models[0].split('/')[-2]}" + \
                              f"_{models[1].split('/')[-1] if models[1].split('/')[-1] != '' else models[1].split('/')[-2]}_{str(args.collocate_percentage).split('.')[-1]}.bin"

        temp = time.time()
        if os.path.exists(args.delta_path):
            added_param = torch.load(args.delta_path)
            print(f"loading existed diff checkpoint to cpu costs {time.time() - temp}\n")
        else:
            model2 = AutoModelForCausalLM.from_pretrained(models[1], torch_dtype=torch.float16,
                                                          trust_remote_code=True, low_cpu_mem_usage=True,
                                                          use_auth_token=args.auth_token)
            print(f"loading {models[1]} to cpu costs {time.time() - temp}\n")

            if type(model) != type(model2):
                print(f"Collocation does not support different model type: {type(model)} and {type(model2)}!")
                return

            if hasattr(model, "model"):
                smart_embedding_resize(
                    model.model.embed_tokens.num_embeddings - model2.model.embed_tokens.num_embeddings,
                    model.model.embed_tokens.num_embeddings, model2)

            # used_names1, graph1, namespace1, n1, e1 = find_submodules_map(
            #     model,
            #     tracing_module_pool=None,
            # )
            # print(f"Traced first model:{models[0]}\n")

            # used_names2, graph2, namespace2, n2, e2 = find_submodules_map(
            # model2,
            # used_names=used_names1,
            # tracing_module_pool=None,
            # )
            # print(f"Traced second model:{models[1]}\n")

            # namespaces = [namespace1, namespace2]
            # nodes = [n1, n2]
            # edges = [e1, e2]
            # graphs = [graph1, graph2]
            # print('Model diff shown below:\n')
            # add_nodes, _, _, _, _= diff_ht_helper(
            #                                         namespaces,
            #                                         nodes,
            #                                         edges,
            #                                         graphs,
            #                                         save_path='./collocation.html',
            #                                         mode='contextual',
            #                                     )
            # added_layers = [used_names2[name] for name in add_nodes]
            # added_param = collections.defaultdict(dict)
            # for k, v in model2.state_dict().items():
            #     for layer in added_layers:
            #         if layer in k:
            #             added_param[layer][k.split(layer + '.')[1]] = v

            model_layers = list(model.state_dict().keys())
            model2_layers = list(model2.state_dict().keys())
            a = [(name, model.state_dict()[name].shape) for name in model_layers]
            b = [(name, model2.state_dict()[name].shape) for name in model2_layers]
            name_map = {
                model2_layers[p[1]]: model_layers[p[0]] for p in lcs_one(a, b, lambda x, y: x == y)
            }
            name_map_keys = list(name_map.keys())
            # This happens in bn layers among torchvision models
            [name_map.pop(k) for k in name_map_keys if "num_batches_tracked" in k]
            added_param = collections.defaultdict(dict)
            for name2, name1 in tqdm(name_map.items()):
                if args.collocate_percentage != 0.:
                    if args.prune_threshold != 0 and "embed" not in name2 and "rotary_emb" not in name2:
                        diff = model2.state_dict()[name2] - model.state_dict()[name1]
                        print(f'before: {torch.count_nonzero(diff) / torch.numel(diff)}')
                        diff[abs(diff) < args.prune_threshold] = 0
                        print(f'after: {torch.count_nonzero(diff) / torch.numel(diff)}')
                        added_param['.'.join(name1.split('.')[:-1])] = (
                        torch.norm(diff, p=2), name1.split('.')[-1], diff)
                    else:
                        l2norm = torch.norm(model2.state_dict()[name2] - model.state_dict()[name1], p=2)
                        added_param['.'.join(name1.split('.')[:-1])] = (
                        l2norm, name1.split('.')[-1], model2.state_dict()[name2])
                    print(f"Different paramsters found in {name2}")

                elif torch.norm(model2.state_dict()[name2] - model.state_dict()[name1], p=2) / torch.norm(
                        model2.state_dict()[name2], 2) > args.diff_threshold:
                    if args.prune_threshold != 0 and "embed" not in name2 and "rotary_emb" not in name2:
                        diff = model2.state_dict()[name2] - model.state_dict()[name1]
                        print(f'before: {torch.count_nonzero(diff) / torch.numel(diff)}')
                        diff[abs(diff) < args.prune_threshold] = 0
                        print(f'after: {torch.count_nonzero(diff) / torch.numel(diff)}')
                        added_param['.'.join(name1.split('.')[:-1])][name1.split('.')[-1]] = diff
                    else:
                        added_param['.'.join(name1.split('.')[:-1])][name1.split('.')[-1]] = model2.state_dict()[name2]
                    print(f"Different paramsters found in {name2}")

            if args.collocate_percentage != 0.:
                sorted_added_param = sorted(added_param.items(), key=lambda x: x[1][0], reverse=True)
                sorted_added_param = sorted_added_param[:int(len(sorted_added_param) * (1 - args.collocate_percentage))]
                added_param = collections.defaultdict(dict)
                for item in sorted_added_param:
                    added_param[item[0]][item[1][1]] = item[1][2]

            if getattr(model2.config, 'rms_norm_eps', None) is not None:
                added_param['rms_norm_eps'] = model2.config.rms_norm_eps

            del model2
            gc.collect()
            torch.save(added_param, args.delta_path)
            print('Recorded model parameter changes\n')

        print(
            f"percentage of diff layers:{len(added_param) / len(model.state_dict()) if getattr(added_param, 'rms_norm_eps', None) is None else (len(added_param) - 1) / len(model.state_dict())}")

        if args.profile_mode == "latency":
            temp = time.time()
            flag = True
            for n, p in tqdm(model.named_modules()):
                if n in added_param and type(p) == torch.nn.modules.linear.Linear:
                    second_weight = added_param[n].get('weight', None)
                    second_bias = added_param[n].get('bias', None)
                    if second_weight is not None and second_bias is not None:
                        if p.bias is not None:
                            new_weight = torch.nn.Parameter(
                                torch.transpose(torch.stack((second_weight, p.weight)), 1, 2))
                            new_bias = torch.nn.Parameter(torch.stack(p.bias, second_bias))

                            def new_forward(self, input: Tensor) -> Tensor:
                                #print(input.shape, input.dtype, '  1')
                                return torch.baddbmm(self.bias,
                                                     input.reshape(2, int((input.size(0) * input.size(1)) / 2),
                                                                   input.size(2)), self.weight).reshape(
                                    input.size(0), input.size(1), self.weight.size(2))

                            bound_method = new_forward.__get__(p, p.__class__)
                            setattr(p, 'weight', new_weight)
                            setattr(p, 'bias', new_bias)
                            setattr(p, 'name_', n)
                            setattr(p, 'forward', bound_method)
                        else:
                            print(f"Incorrect layer match:{n}")
                            return

                    elif second_weight is not None:
                        if args.prune_threshold == 0:
                            new_weight = torch.nn.Parameter(
                                torch.transpose(torch.stack((p.weight, second_weight)), 1, 2))
                            if p.bias is not None:
                                # TODO: edge case can be the bias of second model is simply turned off
                                def new_forward(self, input: Tensor) -> Tensor:
                                    # print(input.shape, input.dtype, '  2')
                                    return torch.baddbmm(self.bias,
                                                         input.reshape(2, int((input.size(0) * input.size(1)) / 2),
                                                                       input.size(2)), self.weight).reshape(
                                        input.size(0), input.size(1), self.weight.size(2))
                            else:
                                # def new_forward(self, input: Tensor) -> Tensor:
                                #     #print(input.shape, input.dtype, '  3')
                                #     return torch.bmm(input.reshape(2, int((input.size(0) * input.size(1)) / 2),
                                #                                    input.size(2)), self.weight).reshape(
                                #         input.size(0), input.size(1), self.weight.size(2))

                                new_weight = torch.nn.Parameter(second_weight)
                                def new_forward(self, input: Tensor) -> Tensor:
                                    input_1, input_2 = torch.tensor_split(input, 2)
                                    out_1 = F.linear(input_1, self.weight)
                                    out_2 = F.linear(input_2, self.second_weight)
                                    return torch.cat((out_1, out_2))


                            bound_method = new_forward.__get__(p, p.__class__)
                            setattr(p, 'second_weight', new_weight)
                            #setattr(p, 'weight', new_weight)
                            setattr(p, 'name_', n)
                            setattr(p, 'forward', bound_method)
                        else:
                            sparse_diff = torch.nn.Parameter(second_weight.to_sparse_csr())
                            if p.bias is not None:
                                def new_forward(self, input: Tensor) -> Tensor:
                                    _, input_2 = torch.tensor_split(input, 2)
                                    diff_out = F.linear(input_2, self.sparse_diff)
                                    out = F.linear(input, self.weight, self.bias)
                                    out[out.shape[0] // 2:, :, :] += diff_out
                                    return out
                            else:
                                def new_forward(self, input: Tensor) -> Tensor:
                                    _, input_2 = torch.tensor_split(input, 2)
                                    diff_out = F.linear(input_2, self.sparse_diff)
                                    out = F.linear(input, self.weight)
                                    out[out.shape[0] // 2:, :, :] += diff_out
                                    return out

                            bound_method = new_forward.__get__(p, p.__class__)
                            setattr(p, 'sparse_diff', sparse_diff)
                            setattr(p, 'name_', n)
                            setattr(p, 'forward', bound_method)

                    elif second_bias is not None:
                        new_weight = torch.transpose(torch.stack((p.weight, p.weight)), 1, 2)
                        if p.bias is not None:
                            def new_forward(self, input: Tensor) -> Tensor:
                                # print(input.shape, input.dtype,'  4')
                                return torch.baddbmm(self.bias,
                                                     input.reshape(2, int((input.size(0) * input.size(1)) / 2),
                                                                   input.size(2)), self.weight).reshape(
                                    input.size(0), input.size(1), self.weight.size(2))

                            setattr(p, 'weight', new_weight)
                        else:
                            print(f"Incorrect layer match:{n}")
                            return

                        bound_method = new_forward.__get__(p, p.__class__)
                        setattr(p, 'name_', n)
                        setattr(p, 'forward', bound_method)

                elif n in added_param and "rotary_emb" in n:
                    inv_freq = added_param[n].get('inv_freq', None)

                    def _set_cos_sin_cache_new(self, seq_len, device, dtype):
                        # print(seq_len, device, dtype,'  5')
                        self.max_seq_len_cached = seq_len
                        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
                        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
                        # Different from paper, but it uses a different permutation in order to obtain the same calculation
                        emb = torch.cat((freqs, freqs), dim=-1).to(device)
                        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
                        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

                        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq_second.dtype)
                        freqs = torch.einsum("i,j->ij", t, self.inv_freq_second)
                        # Different from paper, but it uses a different permutation in order to obtain the same calculation
                        emb = torch.cat((freqs, freqs), dim=-1).to(device)
                        self.register_buffer("cos_cached_second", emb.cos()[None, None, :, :].to(dtype),
                                             persistent=False)
                        self.register_buffer("sin_cached_second", emb.sin()[None, None, :, :].to(dtype),
                                             persistent=False)

                    def new_forward(self, x, seq_len=None):
                        # print(x.shape, seq_len, '    6')
                        if seq_len > self.max_seq_len_cached:
                            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
                        return (
                            [self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                             self.cos_cached_second[:, :, :seq_len, ...].to(dtype=x.dtype)],
                            [self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                             self.sin_cached_second[:, :, :seq_len, ...].to(dtype=x.dtype)]
                        )

                    bound_method_1 = _set_cos_sin_cache_new.__get__(p, p.__class__)
                    bound_method_2 = new_forward.__get__(p, p.__class__)
                    setattr(p, 'inv_freq_second', torch.nn.Parameter(inv_freq))
                    setattr(p, '_set_cos_sin_cache', bound_method_1)
                    setattr(p, 'forward', bound_method_2)
                    setattr(p, 'name_', n)
                    p._set_cos_sin_cache(seq_len=p.max_seq_len_cached, device=p.inv_freq.device,
                                         dtype=p.cos_cached.dtype)

                    if flag:
                        def new_apply_rotary_pos_emb(q, k, cos, sin, position_ids):
                            # print(position_ids.size(0), '    7')
                            if position_ids.size(0) > 1:
                                position_ids_1, position_ids_2 = torch.split(position_ids,
                                                                             (position_ids.size(0) / 2).__ceil__())
                            else:
                                position_ids_1, position_ids_2 = position_ids, position_ids

                            if len(cos) > 1:
                                q_1, q_2 = torch.split(q, (q.size(0) / 2).__ceil__())
                                k_1, k_2 = torch.split(k, (k.size(0) / 2).__ceil__())
                                # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
                                cos_1 = cos[0].squeeze(1).squeeze(0)  # [seq_len, dim]
                                sin_1 = sin[0].squeeze(1).squeeze(0)  # [seq_len, dim]
                                cos_1 = cos_1[position_ids_1].unsqueeze(1)  # [bs, 1, seq_len, dim]
                                sin_1 = sin_1[position_ids_1].unsqueeze(1)  # [bs, 1, seq_len, dim]
                                q_embed_1 = (q_1 * cos_1) + (
                                            transformers.models.llama.modeling_llama.rotate_half(q_1) * sin_1)
                                k_embed_1 = (k_1 * cos_1) + (
                                            transformers.models.llama.modeling_llama.rotate_half(k_1) * sin_1)
                                cos_2 = cos[1].squeeze(1).squeeze(0)  # [seq_len, dim]
                                sin_2 = sin[1].squeeze(1).squeeze(0)  # [seq_len, dim]
                                cos_2 = cos_2[position_ids_2].unsqueeze(1)  # [bs, 1, seq_len, dim]
                                sin_2 = sin_2[position_ids_2].unsqueeze(1)  # [bs, 1, seq_len, dim]
                                q_embed_2 = (q_2 * cos_2) + (
                                            transformers.models.llama.modeling_llama.rotate_half(q_2) * sin_2)
                                k_embed_2 = (k_2 * cos_2) + (
                                            transformers.models.llama.modeling_llama.rotate_half(k_2) * sin_2)
                                return torch.cat((q_embed_1, q_embed_2)), torch.cat((k_embed_1, k_embed_2))
                            else:
                                # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
                                cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
                                sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
                                cos = cos[position_ids_1].unsqueeze(1)  # [bs, 1, seq_len, dim]
                                sin = sin[position_ids_1].unsqueeze(1)  # [bs, 1, seq_len, dim]
                                q_embed = (q * cos) + (transformers.models.llama.modeling_llama.rotate_half(q) * sin)
                                k_embed = (k * cos) + (transformers.models.llama.modeling_llama.rotate_half(k) * sin)
                                return q_embed, k_embed

                        transformers.models.llama.modeling_llama.apply_rotary_pos_emb = new_apply_rotary_pos_emb
                        flag = False

                elif n in added_param and type(p) == torch.nn.modules.sparse.Embedding:
                    second_weight = added_param[n].get('weight', None)

                    def new_forward(self, input: Tensor) -> Tensor:
                        #print(input.shape, input.dtype,'   8')
                        input_1, input_2 = torch.tensor_split(input, 2)
                        return torch.cat(
                            (F.embedding(input_1, self.weight, self.padding_idx, self.max_norm, self.norm_type,
                                         self.scale_grad_by_freq, self.sparse),
                             F.embedding(input_2, self.second_weight, self.padding_idx, self.max_norm, self.norm_type,
                                         self.scale_grad_by_freq, self.sparse)))

                    bound_method = new_forward.__get__(p, p.__class__)
                    setattr(p, 'second_weight', torch.nn.Parameter(second_weight))
                    setattr(p, 'name_', n)
                    setattr(p, 'forward', bound_method)

                elif n in added_param and "llamarmsnorm" in str(type(p)).casefold():
                    second_weight = added_param[n].get('weight', None)
                    variance_epsilon = added_param.get('rms_norm_eps', None)

                    def new_forward(self, hidden_states):
                        # print(hidden_states.shape, hidden_states.dtype, '   9')
                        input_dtype = hidden_states.dtype
                        hidden_states_1, hidden_states_2 = torch.split(hidden_states,
                                                                       (hidden_states.size(0) / 2).__ceil__())
                        variance_1 = hidden_states_1.to(torch.float32).pow(2).mean(-1, keepdim=True)
                        hidden_states_1 = hidden_states_1 * torch.rsqrt(variance_1 + self.variance_epsilon)
                        hidden_states_1 = (self.weight * hidden_states_1).to(input_dtype)
                        variance_2 = hidden_states_2.to(torch.float32).pow(2).mean(-1, keepdim=True)
                        hidden_states_2 = hidden_states_2 * torch.rsqrt(variance_2 + self.second_variance_epsilon)
                        hidden_states_2 = (self.second_weight * hidden_states_2).to(input_dtype)
                        return torch.cat((hidden_states_1, hidden_states_2))

                    if variance_epsilon is None:
                        variance_epsilon = p.variance_epsilon
                    bound_method = new_forward.__get__(p, p.__class__)
                    setattr(p, 'second_weight', torch.nn.Parameter(second_weight))
                    setattr(p, 'second_variance_epsilon', variance_epsilon)
                    setattr(p, 'name_', n)
                    setattr(p, 'forward', bound_method)

            print(
                f"Loading collocated model to cpu costs {time.time() - temp} which can be optimized by multiprocessing\n")
            model = model.to("cuda")
            with torch.no_grad():
                run_collocated_workload(model, args.seq_len, args.b_sizes, args.max_new_tokens)
            return

        else:
            m2_state_dict = {}
            for k, v in added_param.items():
                if k != 'rms_norm_eps':
                    for k_, v_ in added_param[k].items():
                        m2_state_dict[".".join([k, k_])] = v_
            print(m2_state_dict.keys())
            model.load_state_dict(m2_state_dict, strict=False)
            for n, p in model.named_modules():
                if n in added_param and "rotary" in str(type(p)).casefold():
                    p._set_cos_sin_cache(seq_len=p.max_position_embeddings, device=p.inv_freq.device,
                                         dtype=p.cos_cached.dtype)
                elif n in added_param and "layernorm" in str(type(p)).casefold() and getattr(added_param,
                                                                                             'rms_norm_eps',
                                                                                             None) is not None:
                    p.variance_epsilon = added_param['rms_norm_eps']

            model = model.to("cuda")
            tokenizer = transformers.AutoTokenizer.from_pretrained(models[1], trust_remote_code=True,
                                                                   use_auth_token=args.auth_token)
            print_memory_allocated()
            print(f"Running collocated {models[1]}:")
            result = run_test(model, tokenizer, tasks, batch_size, args.test_limit)
            print(json.dumps(result, indent=4))
            return

    elif len(models) == 2 and args.mode == "sequence":

        model2 = AutoModelForCausalLM.from_pretrained(models[1], torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                      trust_remote_code=True, use_auth_token=args.auth_token).to(
            "cuda" if torch.cuda.is_available() else "cpu")
        print_memory_allocated()
        if args.profile_mode == "latency":
            with torch.no_grad():
                run_sequential_workload([model.to("cuda" if torch.cuda.is_available() else "cpu"), model2],
                                        args.seq_len, args.b_sizes, args.max_new_tokens)
            return

        tokenizer = transformers.AutoTokenizer.from_pretrained(models[1], trust_remote_code=True,
                                                               use_auth_token=args.auth_token)
        print_memory_allocated()
        print(f"Running {models[1]}:")
        result = run_test(model2, tokenizer, tasks, batch_size, args.test_limit)
        print(json.dumps(result, indent=4))
        del model2
        clear_torch_cache()

    elif len(models) == 2 and args.mode == "concurrent":

        if args.profile_mode == "latency":
            model2 = AutoModelForCausalLM.from_pretrained(models[1], torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                          trust_remote_code=True, use_auth_token=args.auth_token)
            with torch.no_grad():
                run_concurrent_workload([model, model2], args.seq_len, args.b_sizes, args.max_new_tokens)
            return
        else:
            print("Profiling concurrent models' performance is not supported!")
            return

    else:
        print("Setup not supported!")
        return

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(models[0], trust_remote_code=True,
                                                           use_auth_token=args.auth_token)
    print_memory_allocated()
    print(f"Running {models[0]}:")
    result = run_test(model, tokenizer, tasks, batch_size, args.test_limit)
    result = aggregate_index(result, 0)
    print(json.dumps(result, indent=4))


if __name__ == '__main__':
    mp.set_start_method("spawn")
    args = parser.parse_args()
    main(args)

import os
os.environ['HF_HOME'] = '/workspace/HF_cache/'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/HF_cache/transformers_cache/'
import torch
import time
import sys
MGIT_PATH=os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(MGIT_PATH)
from typing import Any, Callable, Dict, Tuple, Optional
from utils.lineage.graph import *
from utils import meta_functions
import transformers
from transformers import AutoModelForCausalLM
# To control logging level for various modules used in the application:
import logging
import re
import argparse
import shutil
from lm_eval import evaluator

def get_storage_space(path_name):
    str_ret = os.popen("du -s " + path_name).read()
    store_amount = int(str_ret.split("\t")[0])
    return store_amount


def compute_storage_savings(graph):
    return sum(
        get_storage_space(node_name) for node_name in graph.nodes
    ) / get_storage_space(graph.param_home)

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
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

def neg_word_perplexity(model, lineage_dataset, tokenizer):
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    model.device = model.module.device
    model.config = model.module.config
    results = evaluator.simple_evaluate(
            model=model,
            tokenizer=tokenizer,
            tasks=[lineage_dataset.dataset],
            batch_size = 8,
            device= "cuda" if torch.cuda.is_available() else "cpu",
            limit=0.01,
        )
    model = model.module
    clear_torch_cache()
    print({"neg_word_perplexity": -results['results'][f"{lineage_dataset.dataset}_0"]["word_perplexity"]})
    return {"neg_word_perplexity": -results['results'][f"{lineage_dataset.dataset}_0"]["word_perplexity"]}

parser = argparse.ArgumentParser(description="G6 Construction")
parser.add_argument(
    "-g",
    "--g-path",
    default="/workspace/llms",
    type=str,
    help="path containing g7 models",
)
parser.add_argument("--skip_save_models", default=False, action="store_true")
parser.add_argument(
    "--compression_mode", default="lzma", type=str, help="storage compression mode"
)
parser.add_argument("--single_model_compression", default=False, action="store_true")
parser.add_argument("--is_delta", default=True, action="store_false")
parser.add_argument("--no_quantize_delta", default=True, action="store_false", dest="quantize_delta", 
                    help="False means the deltas are not quantizied when compressing (default to True)",)
parser.add_argument("--unique_hash", default=False, action="store_true", dest="unique_hash")

def main(args):
    #set_global_logging_level(logging.ERROR, ["transformers"])
    shutil.rmtree("parameter_store", ignore_errors=True)
    start_time = time.time()

    llama = 'huggyllama/llama-7b'
    llama_2 = "meta-llama/Llama-2-7b-hf"
    llama_2_chat = "meta-llama/Llama-2-7b-chat-hf"
    alpaca = 'tatsu-lab/alpaca-7b/'
    vicuna_v1_1 = 'lmsys/vicuna-7b-v1.1' # 'lmsys/vicuna-7b-delta-v0', 'lmsys/vicuna-7b-v1.3'
    koala = 'TheBloke/koala-7B-HF' # 'young-geng/koala' diff
    alpaca_lora = 'tloen/alpaca-lora-7b'
    pythia = 'EleutherAI/pythia-6.9b/revision/step143000'
    stablelm = 'stabilityai/stablelm-base-alpha-7b'
    dolly_v2 = 'databricks/dolly-v2-7b'
    gptj = 'EleutherAI/gpt-j-6B'
    dolly_v1 = 'databricks/dolly-v1-6b'
    mpt = 'mosaicml/mpt-7b'
    mpt_30b = 'mosaicml/mpt-30b'
    falcon = 'tiiuae/falcon-7b'
    model_pool = [  llama,
                    pythia,
                    stablelm,
                    gptj,
                    mpt,
                    falcon,
                    alpaca,
                    koala,
                    alpaca_lora,
                    dolly_v1,
                    dolly_v2,
                    llama_2,
                    llama_2_chat,
                ]

    def pytorch_init_function(cur_node, parent_list):
        if cur_node.init_checkpoint is None:
            init_checkpoint = cur_node.output_dir
        else:
            init_checkpoint = cur_node.init_checkpoint
        print(f"WARNING: function should only be called once for initilizing {init_checkpoint}") 
        try:
            cur_node.model_inst.model = AutoModelForCausalLM.from_pretrained(   
                                                                                init_checkpoint,
                                                                                trust_remote_code=True,
                                                                                torch_dtype=torch.float16,
                                                                                low_cpu_mem_usage=True,
                                                                            )
            cur_node.model_inst.tokenizer = transformers.AutoTokenizer.from_pretrained(
                                                                                init_checkpoint,
                                                                                torch_dtype=torch.float16,
                                                                                trust_remote_code=True,
                                                                            )
        except Exception as e:
            model_path, revision = re.split(args.g_path + '/|/revision/', init_checkpoint)[-2:]
            cur_node.model_inst.model = AutoModelForCausalLM.from_pretrained(   
                                                                                model_path,
                                                                                revision=revision,
                                                                                trust_remote_code=True,
                                                                                torch_dtype=torch.float16,
                                                                                low_cpu_mem_usage=True,
                                                                            )
            cur_node.model_inst.tokenizer = transformers.AutoTokenizer.from_pretrained(
                                                                                model_path,
                                                                                revision=revision,
                                                                                torch_dtype=torch.float16,
                                                                                trust_remote_code=True,
                                                                            )
        clear_torch_cache()
        print("Successfully loaded {0} with self-defined init_func.".format(init_checkpoint))
    
    
    g = LineageGraph(
        compression_mode=args.compression_mode,
        single_model_compression=args.single_model_compression,
    )
    test = LineageTest(
        eval_dataset=LineageDataset(dataset="pile_arxiv"),
        metric_for_best_model="neg_word_perplexity",
        custom_test_function=neg_word_perplexity,
        name="neg_word_perplexity",
    )
    g.register_test_to_type(test, "llm")

    for model in model_pool:
        print('Inserting: ' + model)  
        node = LineageNode(
                            model_init_function=pytorch_init_function, 
                            #init_checkpoint=model,
                            output_dir=os.path.join(args.g_path, model),
                            task_type="causallm",
                            model_type="llm",
                            trust_remote_code=True,
                            is_delta=args.is_delta,
                            quantize_delta = args.quantize_delta,
                        )
        
        if not g.add(node):
            g.add_root(node)
            print(g.log["adapted"][("root", node.output_dir)])
        if node.get_model().traced is None:
            node.get_model().traced = trace_model(node.get_model().model)
        if not args.skip_save_models:
            node.get_model().save(unique_hash=args.unique_hash)
        for ex_node in g.nodes.values():
            if ex_node is not node:
                ex_node.unload_model(save_model=False)
        print('\n')

    if not args.skip_save_models:
        print(meta_functions.show_result_table(g, show_metrics=True))
        print(f"Storage savings: {compute_storage_savings(g):.3f}")

    print(f"Total time: {(time.time() - start_time) / 3600.0:.3f} hours")
    g.save("./", save_models=False)
    _ = g.show(save_path="./LineageGraph_LLM.html")

    del g
    clear_torch_cache()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

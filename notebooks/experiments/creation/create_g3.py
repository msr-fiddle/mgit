import os
import sys
MGIT_PATH=os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(MGIT_PATH)
from utils.lineage.graph import *
from utils import meta_functions
from tqdm import tqdm
from scripts.create_pruned_models import prune_remove
import torch
import time

torch.manual_seed(0)
import torchvision
from torchvision import transforms
from scripts.create_pruned_models import accuracy
import numpy as np
import shutil
import argparse

import re
import glob

TRANSFORMERS_CACHE = '/workspace/HF_cache/transformers_cache/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="G3 Construction")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-g",
    "--g-path",
    default="/datadrive3/mgit/g3",
    type=str,
    help="path containing g3 models",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument("--num_fl_rounds", default=10, type=int, help="number of FL rounds")
parser.add_argument(
    "--portion", default=1, type=float, help="portion of val data to use"
)
parser.add_argument("--skip_save_models", default=False, action="store_true")
parser.add_argument(
    "--compression_mode", default="lzma", type=str, help="storage compression mode"
)
parser.add_argument("--single_model_compression", default=False, action="store_true")
parser.add_argument("--is_delta", default=True, action="store_false")
parser.add_argument("--no_quantize_delta", default=True, action="store_false", dest="quantize_delta")


def get_folder_size(Folderpath):
    size = 0
    for path, dirs, files in os.walk(Folderpath):
        for f in files:
            fp = os.path.join(path, f)
            size += os.stat(fp).st_size
    return size


def custom_test_function(model, lineage_dataset, tokenizer):
    model.eval()
    dataloader = lineage_dataset.get_dataset()
    acc1s = []
    acc5s = []
    for i, (batch, target) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            output = model(batch.to(device))
            acc1, acc5 = accuracy(output, target.to(device), topk=(1, 5))
            acc1s.append(acc1.cpu().numpy())
            acc5s.append(acc5.cpu().numpy())
    return {"accuracy": np.mean(np.array(acc1s))}


def pytorch_init_function(model_path, cur_node, parent_list):
    print("loading model: ", cur_node.output_dir)
    model = torch.load(model_path, map_location=device)
    cur_node.model_inst.model = model["model"]


def create_node(g, node_name, parents, g3_path, is_delta, skip_save_models, quantize_delta):
    shutil.rmtree(node_name, ignore_errors=True)
    model_path = os.path.join(g3_path, node_name + ".pt")
    init_function = lambda x, y: pytorch_init_function(model_path, x, y)

    node = LineageNode(
        model_init_function=init_function,
        task_type="image_classification",
        output_dir=node_name,
        model_type="torchvision",
        is_delta=is_delta,
        quantize_delta = quantize_delta,
    )
    if len(parents) == 0:
        g.add_root(node)
    else:
        for parent in parents:
            g.add(node, etype="adapted", parent=parent)
    # TODO: Run tests?

    if not skip_save_models:
        g.get_node(node_name).get_model().save()


def main(my_args):
    compression_mode = my_args.compression_mode
    single_model_compression = my_args.single_model_compression

    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")

    start_time = time.time()

    g3_path = my_args.g_path
    imagenet_data = torchvision.datasets.ImageFolder(
        os.path.join(my_args.data, "val"),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    test_size = int(my_args.portion * len(imagenet_data))
    test_dataset, _ = torch.utils.data.random_split(
        imagenet_data, [test_size, len(imagenet_data) - test_size]
    )
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=my_args.batch_size,
        shuffle=True,
        num_workers=my_args.workers,
    )
    lineage_eval_dataset = LineageDataset(dataset=dataloader)

    test1 = LineageTest(
        eval_dataset=lineage_eval_dataset,
        metric_for_best_model="accuracy",
        custom_test_function=custom_test_function,
        name="test1",
    )

    shutil.rmtree("parameter_store", ignore_errors=True)
    g = LineageGraph(
        compression_mode=compression_mode,
        single_model_compression=single_model_compression,
    )
    g.register_test_to_type(test1, "torchvision")

    def pytorch_init_function(model_path, cur_node, parent_list):
        print("\nloading model: ", cur_node.output_dir)
        model = torch.load(model_path, map_location=device)
        cur_node.model_inst.model = prune_remove(model["model"])

    last_master_model_name = None
    all_models = []
    for round_id in range(args.num_fl_rounds):
        worker_id_models_regex = os.path.join(
            g3_path, f"resnet_fl_round={round_id}_worker_id=*"
        )
        matching_filepaths = glob.glob(worker_id_models_regex)
        worker_ids_in_this_round = []
        for matching_filepath in matching_filepaths:
            m = re.match(r".*worker\_id=(\d+)", matching_filepath)
            assert m is not None
            worker_ids_in_this_round.append(int(m.group(1)))
        assert len(worker_ids_in_this_round) > 0
        cur_model_set = []
        for worker_id in worker_ids_in_this_round:
            worker_model_name = f"resnet_fl_round={round_id}_worker_id={worker_id}"
            # Create worker_models for this round, connect to this round's master.
            master_model_names = []
            if last_master_model_name is not None:
                master_model_names = [last_master_model_name]
            create_node(
                g,
                worker_model_name,
                master_model_names,
                g3_path,
                my_args.is_delta,
                my_args.skip_save_models,
                my_args.quantize_delta,
            )
            cur_model_set.append(worker_model_name)
            all_models.append(worker_model_name)

        master_model_name = f"resnet_fl_round={round_id}"
        # Create master_model, connect to this round's worker_models if available.
        create_node(
            g,
            master_model_name,
            cur_model_set,
            g3_path,
            my_args.is_delta,
            my_args.skip_save_models,
            my_args.quantize_delta,
        )
        all_models.append(master_model_name)

        last_master_model_name = master_model_name

        for ex_node in g.nodes.values():
            ex_node.unload_model(save_model=False)

    print(meta_functions.show_result_table(g, show_metrics=True))

    orig_store = 0
    for model_name in all_models:
        orig_store += get_folder_size(model_name)

    global_store = get_folder_size("parameter_store")

    print(f"Storage savings: {orig_store / global_store:.3f}")
    print(f"Total time: {(time.time() - start_time) / 3600.0:.3f} hours")

    g.show(etype="adapted")
    g.save("./", save_models=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

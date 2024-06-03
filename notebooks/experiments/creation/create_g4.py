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

TRANSFORMERS_CACHE = '/workspace/HF_cache/transformers_cache/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="G4 Construction")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-g",
    "--g-path",
    default="/datadrive3/mgit/g4",
    type=str,
    help="path containing g4 models",
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


def top1_accuracy(model, lineage_dataset, tokenizer):
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


def sparsity(model, lineage_dataset, tokenizer):
    model.eval()
    n_param = 0
    nnz_param = 0
    pruned_param = set()
    for name, module in model.named_modules():
        for _module in [torch.nn.Conv2d, torch.nn.Linear]:
            if isinstance(module, _module):
                pruned_param.add(name + ".weight")

    for name, param in model.state_dict().items():
        if name in pruned_param:
            n_param += torch.numel(param)
            nnz_param += torch.count_nonzero(param)

    return {"sparsity": (n_param - nnz_param) * 100 / n_param}


def main(my_args):
    compression_mode = my_args.compression_mode
    single_model_compression = my_args.single_model_compression
    quantize_delta = my_args.quantize_delta

    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")

    start_time = time.time()

    g4_path = my_args.g_path
    densenet = [
        "densenet121_pruned_global_1",
        "densenet121_pruned_global_2",
        "densenet121_pruned_global_3",
        "densenet121_pruned_global_4",
        "densenet121_pruned_global_4_finetuned_best",
        "densenet121_pruned_global_5",
        "densenet121_pruned_global_5_finetuned_best",
    ]
    mobilenet = [
        "mobilenet_v3_pruned_global_1",
        "mobilenet_v3_pruned_global_2",
        "mobilenet_v3_pruned_global_3",
        "mobilenet_v3_pruned_global_3_finetuned_best",
        "mobilenet_v3_pruned_global_4",
        "mobilenet_v3_pruned_global_4_finetuned_best",
        "mobilenet_v3_pruned_global_5",
        "mobilenet_v3_pruned_global_5_finetuned_best",
    ]
    resnet = [
        "resnet50_pruned_global_1",
        "resnet50_pruned_global_2",
        "resnet50_pruned_global_3",
        "resnet50_pruned_global_4",
        "resnet50_pruned_global_4_finetuned_best",
        "resnet50_pruned_global_5",
        "resnet50_pruned_global_5_finetuned_best",
    ]
    model_pool = {"mobilenet": mobilenet, "resnet": resnet, "densenet": densenet}
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
        custom_test_function=top1_accuracy,
        name="top1_acc",
    )

    test2 = LineageTest(
        eval_dataset=LineageDataset(dataset=""),
        metric_for_best_model="sparsity",
        custom_test_function=sparsity,
        name="sparsity",
    )

    shutil.rmtree("parameter_store", ignore_errors=True)
    g = LineageGraph(
        compression_mode=compression_mode,
        single_model_compression=single_model_compression,
    )
    g.register_test_to_type(test1, "torchvision")
    g.register_test_to_type(test2, "torchvision")

    last_name = None
    last_arch = None
    is_delta = my_args.is_delta

    for arch in model_pool:
        for model_name in model_pool[arch]:
            shutil.rmtree(model_name, ignore_errors=True)
            print("\ninserting: ", model_name)
            model_path = os.path.join(g4_path, model_name + ".pt")
            if "_finetuned_best" in model_name:

                def pytorch_init_function(cur_node, parent_list):
                    print("loading model: ", cur_node.output_dir)
                    model = torch.load(model_path, map_location=device)
                    cur_node.model_inst.model = prune_remove(model["model"])

            else:

                def pytorch_init_function(cur_node, parent_list):
                    print("loading model: ", cur_node.output_dir)
                    model = torch.load(model_path, map_location=device)
                    cur_node.model_inst.model = prune_remove(model)

            if arch != last_arch:
                node = LineageNode(
                    model_init_function=pytorch_init_function,
                    task_type="image_classification",
                    output_dir=model_name,
                    model_type="torchvision",
                    is_delta=is_delta,
                    quantize_delta = quantize_delta,
                )
                g.add_root(node)
                if not single_model_compression:
                    g.get_node(model_name).run_all_tests()
                last_name = model_name
                last_arch = arch
            else:
                node = LineageNode(
                    model_init_function=pytorch_init_function,
                    task_type="image_classification",
                    output_dir=model_name,
                    model_type="torchvision",
                    is_delta=is_delta,
                    quantize_delta = quantize_delta,
                )
                if "_finetuned_best" in model_name:
                    g.add(
                        node,
                        etype="adapted",
                        parent=model_name.replace("_finetuned_best", ""),
                    )
                else:
                    g.add(node, etype="adapted", parent=last_name)
                    last_name = model_name
                # Run tests if is_delta=False and single_model_compression=False since they haven't run yet.
                if not is_delta and not single_model_compression:
                    g.get_node(model_name).run_all_tests()

            if not my_args.skip_save_models:
                g.get_node(model_name).get_model().save()
            for ex_node in g.nodes.values():
                ex_node.unload_model(save_model=False)

    print(meta_functions.show_result_table(g, show_metrics=True))

    orig_store = 0
    for arch in model_pool.keys():
        for model_name in model_pool[arch]:
            orig_store += get_folder_size(model_name)

    global_store = get_folder_size("parameter_store")

    print(f"Storage savings: {orig_store / global_store:.3f}")
    print(f"Total time: {(time.time() - start_time) / 3600.0:.3f} hours")

    g.show(etype="adapted")
    g.save("./", save_models=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

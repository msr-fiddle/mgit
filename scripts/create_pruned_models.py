import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils import prune
from torchvision.models import densenet121, mobilenet_v3_large, resnet50

import numpy as np


def get_model(model_name):
    if model_name == "resnet50":
        model = resnet50(pretrained=True)
    elif model_name == "densenet121":
        model = densenet121(pretrained=True)
    elif model_name == "mobilenet_v3":
        model = mobilenet_v3_large(pretrained=True)
    else:
        raise Exception("Invalid model!")
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def prune_model_l1_unstructured(model, prune_dict = {torch.nn.Conv2d:0.1, torch.nn.Linear:0.2}, val_portion=0.1):
    print(f"Pruning model ...")
    if isinstance(model, str):
        model = get_model(model)
    acc1, acc5 = validate_model(model, val_portion)
    print("Accuracy before pruning: {}".format({f"Accuracy@1: {acc1:.3f}, Accuracy@5: {acc5:.3f}"}))

    for name, module in model.named_modules():
        for _module in prune_dict.keys():
            if isinstance(module, _module):
                prune.l1_unstructured(module, name="weight", amount=prune_dict[_module])

    acc1, acc5 = validate_model(model, val_portion)
    print("Accuracy after pruning: {}".format({f"Accuracy@1: {acc1:.3f}, Accuracy@5: {acc5:.3f}"}))
    return model


def prune_model_global_unstructured(model, amount, prune_list = [torch.nn.Conv2d, torch.nn.Linear], val_portion=0.1):
    print(f"Pruning model ...")
    if isinstance(model, str):
        model = get_model(model)
    acc1, acc5 = validate_model(model, val_portion)
    print("Accuracy before pruning: {}".format({f"Accuracy@1: {acc1:.3f}, Accuracy@5: {acc5:.3f}"}))
    
    module_tups = []
    for name, module in model.named_modules():
        for _module in prune_list:
            if isinstance(module, _module):
                module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=amount
    )
    
    acc1, acc5 = validate_model(model, val_portion)
    print("Accuracy after pruning: {}".format({f"Accuracy@1: {acc1:.3f}, Accuracy@5: {acc5:.3f}"}))
    
    return model


def prune_model_l2_structured(model, prune_dict = {torch.nn.Conv2d:0.1}, val_portion=0.1):
    print(f"Pruning model ...")
    if isinstance(model, str):
        model = get_model(model)
    acc1, acc5 = validate_model(model, val_portion)
    print("Accuracy before pruning: {}".format({f"Accuracy@1: {acc1:.3f}, Accuracy@5: {acc5:.3f}"}))
    
    for name, module in model.named_modules():
        for _module in prune_dict.keys():
            if isinstance(module, _module):
                prune.ln_structured(module, 'weight', prune_dict[_module], n=2, dim=1)

    acc1, acc5 = validate_model(model, val_portion)
    print("Accuracy after pruning: {}".format({f"Accuracy@1: {acc1:.3f}, Accuracy@5: {acc5:.3f}"}))
    return model


def prune_remove(model, prune_list = [torch.nn.Conv2d, torch.nn.Linear]):
    for name, module in model.named_modules():
        for _module in prune_list:
            if isinstance(module, _module):
                prune.remove(module, 'weight')
    return model


def quantize_model(model, filepath, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8, val_portion=0.1):
    print(f"Quantize model ...")
    model = get_model(model)
    acc1, acc5 = validate_model(model, val_portion)
    print("Accuracy before quantization: {}".format({f"Accuracy@1: {acc1:.3f}, Accuracy@5: {acc5:.3f}"}))

    model_dynamic_quantized = torch.quantization.quantize_dynamic(
        model, qconfig_spec=qconfig_spec, dtype=torch.qint8)
    
    torch.save(model_dynamic_quantized.state_dict(), filepath)

    acc1, acc5 = validate_model(model, val_portion)
    print("Accuracy after quantization: {}".format({f"Accuracy@1: {acc1:.3f}, Accuracy@5: {acc5:.3f}"}))
    return model


def validate_model(model, portion=0.1):
    torch.manual_seed(0)
    imagenet_data = torchvision.datasets.ImageFolder(
        "/datadrive3/mgit/data/imagenet/val",
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
    test_size = int(portion * len(imagenet_data))
    test_dataset, _ = torch.utils.data.random_split(imagenet_data, [test_size, len(imagenet_data) - test_size])
    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=True, num_workers=4
    )
        
    acc1s = []
    acc5s = []
    for i, (batch, target) in enumerate(dataloader):
        output = model(batch)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1s.append(acc1.numpy())
        acc5s.append(acc5.numpy())

    return np.mean(np.array(acc1s)), np.mean(np.array(acc5s))


ths={'resnet':71.25,
'densenet':70,
'mobilenet':63.625}


def trainable_global_pruned_model(model, amount, prune_list = [torch.nn.Conv2d, torch.nn.Linear], path='pruned_model.pt', threshold=None, val_portion=0.1):
    print(f"Pruning model ...")
    if isinstance(model, str):
        model = get_model(model)
    acc1, acc5 = validate_model(model, val_portion)
    print("Accuracy before pruning: {}".format({f"Accuracy@1: {acc1:.3f}, Accuracy@5: {acc5:.3f}"}))
    
    module_tups = []
    for name, module in model.named_modules():
        for _module in prune_list:
            if isinstance(module, _module):
                module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=amount
    )
    
    acc1, acc5 = validate_model(model, val_portion)
    print("Accuracy after pruning: {}".format({f"Accuracy@1: {acc1:.3f}, Accuracy@5: {acc5:.3f}"}))
    
    if threshold:
        if acc1 >= threshold:
            torch.save(model, path)
        else:
            print(f"Pruning fail: top-1 accuracy lower than {threshold:.3f}")
    else:
        for arch, th in ths:
            if arch in str(model).casefold():
                if acc1 >= th:
                    torch.save(model, path)
                else:
                    print(f"Pruning fail: top-1 accuracy lower than {th:.3f}")
                break
    return model

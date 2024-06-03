# +
import argparse
import copy
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import ResNet50_Weights


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs-per-fl-round', default=3, type=int, metavar='N',
                    help='number of total epochs to run per round')
parser.add_argument('--num-fl-rounds', default=10, type=int,
                    help='number of FL rounds to run')
parser.add_argument('--num-fl-workers', default=5, type=int,
                    help='number of FL workers to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', type=str, required=True,
                    help='path to save checkpoint')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='fine tune pre-trained model')
parser.add_argument("--local_rank", type=int, default=0)

best_prec1 = 0

def print_rank0(message):
    if torch.distributed.get_rank() == 0:
        print(message)

def average_state_dicts(local_weight_state_dicts):
    """
    Returns the average of the weights.
    """
    average_state_dict = copy.deepcopy(local_weight_state_dicts[0])
    n = len(local_weight_state_dicts)
    for key in average_state_dict.keys():
        for i in range(1, n):
            average_state_dict[key] += local_weight_state_dicts[i][key]
        average_state_dict[key] = torch.div(average_state_dict[key], n)
    return average_state_dict

def main(my_args):
    global args, best_prec1
    args = my_args

    # Set up GPU training and call init_process_group.
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Seed random number generators for determinism.
    torch.manual_seed(42)
    random.seed(42)

    # Create model and load default weights.
    print_rank0("=> Creating resnet50 model")
    global_model = models.__dict__['resnet50'](weights=ResNet50_Weights.IMAGENET1K_V1)

    if torch.cuda.is_available():
        global_model.cuda(args.local_rank)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            try:
                if 'epoch' in checkpoint and 'best_prec1' in checkpoint and 'state_dict' in checkpoint:
                    args.start_epoch = checkpoint['epoch']
                    best_prec1 = checkpoint['best_prec1']
                    global_model.load_state_dict(checkpoint['state_dict'])
                    print("=> Loaded checkpoint '{}' (epoch {})"
                          .format(args.evaluate, checkpoint['epoch']))
                else:
                    global_model.load_state_dict(checkpoint)
            except:
                global_model = checkpoint
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    for i in range(args.num_fl_rounds):
        print_rank0(f"=> Starting round {i}...")
        local_models = []
        random_worker_ids = random.sample(list(range(40)), args.num_fl_workers)
        for worker_id in random_worker_ids:
            local_models.append(train_round(global_model, args=args, worker_id=worker_id, round_id=i).cpu())
        average_state_dict = average_state_dicts(
            [local_model.module.state_dict() for local_model in local_models])
        global_model.load_state_dict(average_state_dict)
        save_checkpoint({
            'arch': 'resnet50',
            'model': global_model,
        }, worker_id=None, filename=args.save.replace(".pt", f"_round={i}.pt"))
        print_rank0("=> Averaged local models!")
        validate_on_full_dataset(global_model,
                                 message=f'global (full), round {i}')


def train_round(global_model, args, worker_id, round_id):
    global best_prec1

    print_rank0(f'==> Starting FL training round for worker {worker_id}...')
    local_model = copy.deepcopy(global_model)

    traindir = os.path.join(args.data, f'worker{worker_id}', 'train')
    valdir = os.path.join(args.data, f'worker{worker_id}', 'val')

    # Set up training and validation datasets.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir, transforms.Compose([
            transforms.RandomCrop(224, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # Set up distributed training on GPUs if available.
    if torch.cuda.is_available():
        local_model.cuda(args.local_rank)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        world_size = torch.distributed.get_world_size()
        batch_size = int(args.batch_size / world_size)
        local_model = torch.nn.parallel.DistributedDataParallel(local_model, device_ids=[args.local_rank],
                                                                output_device=args.local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
        batch_size = args.batch_size

    # Set up data loaders.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)

    # Define loss function (criterion) and optimizer.
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       local_model.parameters()),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs_per_fl_round):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, local_model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, local_model, criterion,
                         message=f'worker {worker_id}, round {round_id}')

        # remember best prec@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'model': local_model.module,
        }, worker_id=worker_id, filename=args.save.replace(".pt", f"_round={round_id}.pt"))

    return local_model

def validate_on_full_dataset(model, message):
    global args

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Validate on full dataset.
    valdir = os.path.join(args.data, 'val')
    val_dataset = datasets.ImageFolder(
        valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)

    return validate(val_loader, model, criterion, message)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('===> Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, message):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print_rank0(' * [{message}] Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(message=message, top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, worker_id, filename):
    if worker_id is not None:
        filename = filename.replace(".pt", f"_worker_id={worker_id}.pt")
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

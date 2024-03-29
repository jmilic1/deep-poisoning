import time
import datetime

import numpy as np
import PIL.Image as Image
from tqdm import tqdm

from torch import nn
from other_cleanser import Back
import torch.optim as optim
from torchvision.utils import save_image
import torch

import config
import os
import sys

from train_on_poisoned_set import get_poison_transform
from utils import supervisor, tools, setup_seed
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader


def generate_dataloader(dataset_path='./data', batch_size=128, split='train', shuffle=True,
                        drop_last=False, data_transform=None):
    if data_transform is None:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ])
    dataset_path = os.path.join(dataset_path, 'cifar10')
    if split == 'train':
        train_data = datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=data_transform)
        train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle,
                                       drop_last=drop_last, num_workers=4, pin_memory=True)
        return train_data_loader
    elif split == 'std_test' or split == 'full_test':
        test_data = datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=data_transform)
        test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=shuffle,
                                      drop_last=drop_last, num_workers=4, pin_memory=True)
        return test_data_loader
    elif split == 'valid' or split == 'val':
        val_set_dir = os.path.join('clean_set', 'cifar10', 'clean_split')
        val_set_img_dir = os.path.join(val_set_dir, 'data')
        val_set_label_path = os.path.join(val_set_dir, 'clean_labels')
        val_set = setup_seed.IMG_Dataset(data_dir=val_set_img_dir, label_path=val_set_label_path,
                                         transforms=data_transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle,
                                                 drop_last=drop_last, num_workers=4, pin_memory=True)
        return val_loader
    elif split == 'test':
        test_set_dir = os.path.join('clean_set', 'cifar10', 'test_split')
        test_set_img_dir = os.path.join(test_set_dir, 'data')
        test_set_label_path = os.path.join(test_set_dir, 'labels')
        test_set = setup_seed.IMG_Dataset(data_dir=test_set_img_dir, label_path=test_set_label_path,
                                          transforms=data_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True,
                                                  drop_last=drop_last, num_workers=4, pin_memory=True)
        return test_loader


class BackdoorDefense:
    def __init__(self, args: config.Arguments):
        self.dataset = args.dataset
        if args.no_normalize:
            self.data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor()
            ])
            self.data_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            self.normalizer = transforms.Compose([])
            self.denormalizer = transforms.Compose([])
        else:
            self.data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            self.data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            self.normalizer = transforms.Compose([
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            self.denormalizer = transforms.Compose([
                transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                     [1 / 0.247, 1 / 0.243, 1 / 0.261])
            ])

        self.img_size = 32
        self.num_classes = 10
        self.input_channel = 3
        self.shape = torch.Size([3, 32, 32])

        self.poison_type = args.poison_type
        self.poison_rate = args.poison_rate
        self.cover_rate = args.cover_rate
        self.alpha = args.alpha
        self.trigger = args.trigger
        self.target_class = args.target_class
        self.device = 'cuda'

        self.poison_transform = get_poison_transform(poison_type=args.poison_type, dataset=dataset,
                                                     target_class=args.target_class,
                                                     trigger_transform=self.data_transform,
                                                     alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                     trigger_name=args.trigger_name)
        self.poison_set_dir = supervisor.get_poison_set_dir(dataset, args)


def unpack_poisoned_train_set(dataset, args: config.Arguments, batch_size=128, shuffle=False, data_transform=None):
    """
    Return with `poison_set_dir`, `poisoned_set_loader`, `poison_indices`, and `cover_indices` if available
    """

    if data_transform is None:
        if args.no_normalize:
            data_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])

    poison_set_dir = supervisor.get_poison_set_dir(dataset, args)

    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    cover_indices_path = os.path.join(poison_set_dir, 'cover_indices')  # for adaptive attacks

    poisoned_set = setup_seed.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                          label_path=poisoned_set_label_path, transforms=data_transform)

    poisoned_set_loader = torch.utils.data.DataLoader(poisoned_set, batch_size=batch_size, shuffle=shuffle,
                                                      num_workers=4, pin_memory=True)

    poison_indices = torch.load(poison_indices_path)

    if ('adaptive' in args.poison_type) or args.poison_type == 'TaCT':
        cover_indices = torch.load(cover_indices_path)
        return poison_set_dir, poisoned_set_loader, poison_indices, cover_indices

    return poison_set_dir, poisoned_set_loader, poison_indices, []


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str = None, fmt: str = ':f'):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.name: str = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# region ABL


class ABL(BackdoorDefense):
    """
    Anti-Backdoor Learning

    Args:
        isolation_epochs (int): the number of isolation epochs for backdoor isolation. Default: 20.
        isolation_ratio (float): percentage of inputs to isolate. Default: 0.01.
        gradient_ascent_type (str): 'LGA' (Local Gradient Ascent) or 'Flooding'. Default: 'Flooding'.
        gamma (float): hyperparam for LGA. Default: 0.5.
        flood (float): hyperparam for Flooding. Default: 0.5.
        do_isolate (bool): do isolation phase or not. Default: True.
        finetuning_ascent_model (bool): finetune to raise clean accuracy or not. Default: True.
        finetuning_epochs (int): the number of finetuning epochs. Default: 60.
        unlearning_epochs (int): the number of unlearning epochs. Default: 20.
        do_unlearn (bool): do unlearning phase or not. Default: True.


    .. _Anti-Backdoor Learning:
        https://arxiv.org/abs/2110.11571


    .. _original source code:
        https://github.com/bboylyg/ABL

    """

    def __init__(self, args: config.Arguments,
                 isolation_epochs=20, isolation_ratio=0.01, gradient_ascent_type='Flooding', gamma=0.5, flooding=0.5,
                 do_isolate=True,
                 finetuning_ascent_model=True, finetuning_epochs=60, unlearning_epochs=5, lr_unlearning=5e-4,
                 do_unlearn=True):
        super().__init__(args)

        self.args = args
        self.isolation_epochs = isolation_epochs
        self.isolation_ratio = isolation_ratio
        self.gradient_ascent_type = gradient_ascent_type
        self.gamma = gamma
        self.flooding = flooding
        self.finetuning_ascent_model = finetuning_ascent_model
        self.finetuning_epochs = finetuning_epochs
        self.unlearning_epochs = unlearning_epochs

        self.save_interval = 10

        self.tuning_lr = 0.1
        self.lr_finetuning_init = 0.1
        self.batch_size_isolation = 64
        self.batch_size_finetuning = 64
        self.batch_size_unlearning = 64

        self.lr_unlearning = lr_unlearning

        self.do_isolate = do_isolate
        self.do_unlearn = do_unlearn

        # self.tf_compose_isolation = transforms.Compose([
        #     transforms.Resize((32, 32)),
        #     transforms.ToTensor(),
        #     # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        #     # transforms.RandomCrop(32, padding=4),
        #     # transforms.RandomHorizontalFlip(),
        #     # Cutout(1, 3)
        # ])

        # self.tf_compose_finetuning = transforms.Compose([
        #     transforms.Resize((32, 32)),
        #     transforms.ToTensor(),
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     Cutout(1, 3),
        #     # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # ])

        # self.tf_compose_unlearning = transforms.Compose([
        #     transforms.Resize((32, 32)),
        #     transforms.ToTensor(),
        #     # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # ])

        self.tf_compose_isolation = self.data_transform
        self.tf_compose_finetuning = self.data_transform_aug
        self.tf_compose_unlearning = self.data_transform

        self.folder_path = 'other_defenses_tool_box/results/ABL'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)

        self.test_loader = generate_dataloader(batch_size=100,
                                               split='std_test',
                                               shuffle=False,
                                               drop_last=False,
                                               data_transform=self.tf_compose_isolation)
        # data_transform=self.data_transform)
        # data_transform=transforms.Compose([transforms.ToTensor()]))
        # self.args.no_normalize = True

        if gradient_ascent_type == 'Flooding':
            print(f"Gradient ascent method: 'Flooding', `flooding` = {flooding}")
        elif gradient_ascent_type == 'LGA':
            print(f"Gradient ascent method: 'LGA', `gamma` = {gamma}")

    def detect(self):
        if self.do_isolate: self.isolate()
        if self.do_unlearn: self.unlearn()

    def isolate(self):
        """
        ABL Step 1: Isolate 1% inputs with the lowest loss. The training process is enhanced with LGA.
        """
        print('----------- Train isolated model -----------')
        ascent_model = self.train_isolation()

        print('----------- Calculate loss value per example -----------')
        losses_idx = self.compute_loss_value(ascent_model)

        print('----------- Collect isolation data -----------')
        self.isolate_data(losses_idx)

        # # test model and save
        # result_file = os.path.join(self.folder_path, 'FP_%s.pt' % supervisor.get_dir_core(self.args))
        # torch.save(self.model.state_dict(), result_file)
        # print('Ascent Model Saved at:', result_file)
        # val_atk(self.args, self.model)

    def unlearn(self):
        """
        ABL Step 2: Unlearn backdoor task with GGA
        """
        self.train_unlearn()

    def compute_loss_value(self, model_ascent):
        args = self.args
        # Calculate loss value per example
        # Define loss function
        # criterion = nn.CrossEntropyLoss().cuda()
        criterion = nn.CrossEntropyLoss(reduction='none').cuda()

        model_ascent.eval()
        losses_record = []

        # poison_set_dir, poisoned_set_loader, poison_indices, cover_indices = unpack_poisoned_train_set(args, batch_size=1, shuffle=False, data_transform=self.tf_compose_isolation)
        poison_set_dir, poisoned_set_loader, poison_indices, cover_indices = unpack_poisoned_train_set(dataset, args,
                                                                                                       batch_size=128,
                                                                                                       shuffle=False,
                                                                                                       data_transform=self.tf_compose_isolation)

        for img, target in tqdm(poisoned_set_loader):
            img = img.cuda()
            target = target.cuda()

            with torch.no_grad():
                output = model_ascent(img)
                loss = criterion(output, target)
                # print(loss)

            # losses_record.append(loss.item())
            losses_record.append(loss)

        losses_record = torch.cat(losses_record, dim=0).tolist()
        losses_idx = np.argsort(np.array(losses_record))  # get the index of examples by loss value in ascending order

        # Show the lowest 10 loss values
        losses_record_arr = np.array(losses_record)
        print('Smallest 10 loss values:', losses_record_arr[losses_idx[:10]])
        print('Smallest 10 loss indices:', losses_idx[:10])

        return losses_idx

    def isolate_data(self, losses_idx):
        args = self.args

        # Initialize lists
        other_examples = []
        isolation_examples = []

        cnt = 0
        ratio = self.isolation_ratio

        poison_set_dir, poisoned_set_loader, poison_indices, cover_indices = unpack_poisoned_train_set(dataset, args,
                                                                                                       batch_size=1,
                                                                                                       shuffle=False,
                                                                                                       data_transform=self.tf_compose_isolation)
        # print('full_poisoned_data_idx:', len(losses_idx))
        perm = losses_idx[0: int(len(losses_idx) * ratio)]
        isolation_indices = losses_idx[0:int(len(losses_idx) * ratio)].tolist()
        other_indices = losses_idx[int(len(losses_idx) * ratio):].tolist()

        # save the isolation indices
        data_path_isolation = os.path.join(self.folder_path, 'abl_%s_isolation_ratio=%.3f_examples_indices_seed=%d' % (
            supervisor.get_dir_core(self.args), self.isolation_ratio, self.args.seed))
        data_path_other = os.path.join(self.folder_path, 'abl_%s_other_ratio=%.3f_examples_indices_seed=%d' % (
            supervisor.get_dir_core(self.args), self.isolation_ratio, self.args.seed))
        torch.save(isolation_indices, data_path_isolation)
        torch.save(other_indices, data_path_other)

        prec_isolation = 1 - len(set(isolation_indices) - set(poison_indices) - set(cover_indices)) / len(
            isolation_indices)
        print('Finish collecting {} isolation examples (Prec: {:.6f}), saved at \'{}\''.format(len(isolation_indices),
                                                                                               prec_isolation,
                                                                                               data_path_isolation))
        print('Finish collecting {} other examples, saved at \'{}\''.format(len(other_indices), data_path_other))

    def train_step_isolation(self, train_loader, model_ascent, optimizer, criterion, epoch):
        args = self.args

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model_ascent.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            if self.gradient_ascent_type == 'LGA':
                output = model_ascent(img)
                loss = criterion(output, target)
                # add Local Gradient Ascent(LGA) loss
                loss_ascent = torch.sign(loss - self.gamma) * loss
                # loss_ascent = loss

            elif self.gradient_ascent_type == 'Flooding':
                output = model_ascent(img)
                # output = student(img)
                loss = criterion(output, target)
                # add flooding loss
                loss_ascent = (loss - self.flooding).abs() + self.flooding
                # loss_ascent = loss

            else:
                raise NotImplementedError

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss_ascent.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            loss_ascent.backward()
            optimizer.step()

        print('\nEpoch[{0}]: '
              'loss: {losses.avg:.4f}  '
              'prec@1: {top1.avg:.2f}  '
              'prec@5: {top5.avg:.2f}  '
              'lr: {lr:.4f}  '.format(epoch, losses=losses, top1=top1, top5=top5, lr=optimizer.param_groups[0]['lr']))

    def train_step_finetuing(self, train_loader, model_ascent, optimizer, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model_ascent.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            output = model_ascent(img)

            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('\nEpoch[{0}]: '
              'loss: {losses.avg:.4f}  '
              'prec@1: {top1.avg:.2f}  '
              'prec@5: {top5.avg:.2f}  '
              "lr: {lr:.4f}  ".format(epoch, losses=losses, top1=top1, top5=top5, lr=optimizer.param_groups[0]['lr']))

    def train_step_unlearning(self, train_loader, model_ascent, optimizer, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model_ascent.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            output = model_ascent(img)

            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            (-loss).backward()  # Gradient ascent training
            optimizer.step()

        print('\nEpoch[{0}]: '
              'loss: {losses.avg:.4f}  '
              'prec@1: {top1.avg:.2f}  '
              'prec@5: {top5.avg:.2f}  '
              'lr: {lr:.4f}  '.format(epoch, losses=losses, top1=top1, top5=top5, lr=optimizer.param_groups[0]['lr']))

    def test(self, model, criterion, epoch):
        args = self.args

        test_process = []
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.eval()

        for idx, (img, target) in enumerate(self.test_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            with torch.no_grad():
                output = model(img)
                loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_clean = [top1.avg, top5.avg, losses.avg]

        print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))

        if self.args.poison_type != 'none':
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for idx, (img, target) in enumerate(self.test_loader, start=1):
                img, target = img.cuda(), target.cuda()
                img = img[target != self.target_class]
                target = target[target != self.target_class]
                poison_img, poison_target = self.poison_transform.transform(img, target)

                with torch.no_grad():
                    poison_output = model(poison_img)
                    loss = criterion(poison_output, poison_target)

                prec1, prec5 = accuracy(poison_output, poison_target, topk=(1, 5))
                losses.update(loss.item(), img.size(0))
                top1.update(prec1.item(), img.size(0))
                top5.update(prec5.item(), img.size(0))

            acc_bd = [top1.avg, top5.avg, losses.avg]

            print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))

            return acc_clean, acc_bd
        return acc_clean, acc_clean

    def train_isolation(self):
        args = self.args

        # Load models
        print('----------- Network Initialization --------------')
        # arch = config.arch['abl']
        # model_ascent = arch(depth=16, num_classes=self.num_classes, widen_factor=1, dropRate=0)
        arch = args.arch
        model_ascent = arch(num_classes=self.num_classes)
        model_ascent = model_ascent.cuda()
        print('finished model init...')

        # initialize optimizer
        optimizer = torch.optim.SGD(model_ascent.parameters(),
                                    lr=self.tuning_lr,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)

        # define loss functions
        criterion = nn.CrossEntropyLoss().cuda()

        print('----------- Data Initialization --------------')
        poison_set_dir, poisoned_set_loader, poison_indices, cover_indices = unpack_poisoned_train_set(dataset, args,
                                                                                                       batch_size=self.batch_size_isolation,
                                                                                                       shuffle=True,
                                                                                                       data_transform=self.tf_compose_isolation)

        print('----------- Train Initialization --------------')
        for epoch in range(0, self.isolation_epochs):
            self.adjust_learning_rate(optimizer, epoch)

            # train every epoch
            # if epoch == 0:
            #     # before training test firstly
            #     # val_atk(self.args, model_ascent)
            #     test(model_ascent, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))

            self.train_step_isolation(poisoned_set_loader, model_ascent, optimizer, criterion, epoch)

            # evaluate on testing set
            # val_atk(self.args, model_ascent)
            supervisor.test(model_ascent, test_loader=self.test_loader, poison_test=True,
                            poison_transform=self.poison_transform,
                            num_classes=self.num_classes, source_classes=self.source_classes)

        # save isolated model
        self.save_checkpoint({
            'epoch': self.isolation_epochs,
            'state_dict': model_ascent.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, self.isolation_epochs, True, phase='isolation')

        return model_ascent

    def train_unlearn(self):
        args = self.args

        # Load models
        print('----------- Network Initialization --------------')
        # arch = config.arch['abl']
        # model_ascent = arch(depth=16, num_classes=self.num_classes, widen_factor=1, dropRate=0)
        arch = args.arch
        model_ascent = arch(num_classes=self.num_classes)
        self.load_checkpoint(model=model_ascent,
                             filepath=os.path.join(self.folder_path, 'abl_%s_isolation_epoch=%d_seed=%d.tar' % (
                                 supervisor.get_dir_core(self.args), self.isolation_epochs, self.args.seed)))
        model_ascent = model_ascent.cuda()
        print('Loaded ascent model (isolation)!')

        # initialize optimizer
        optimizer = torch.optim.SGD(model_ascent.parameters(),
                                    lr=0.1,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)

        # define loss functions
        criterion = nn.CrossEntropyLoss().cuda()

        print('----------- Data Initialization --------------')
        data_path_isolation = os.path.join(self.folder_path, 'abl_%s_isolation_ratio=%.3f_examples_indices_seed=%d' % (
            supervisor.get_dir_core(self.args), self.isolation_ratio, self.args.seed))
        data_path_other = os.path.join(self.folder_path, 'abl_%s_other_ratio=%.3f_examples_indices_seed=%d' % (
            supervisor.get_dir_core(self.args), self.isolation_ratio, self.args.seed))
        isolation_indices = torch.load(data_path_isolation)
        other_indices = torch.load(data_path_other)

        # load indices
        poison_set_dir = supervisor.get_poison_set_dir(args)
        poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
        poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')

        # load data
        isolate_poisoned_data_tf = Subset(setup_seed.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                                                 label_path=poisoned_set_label_path,
                                                                 transforms=self.tf_compose_unlearning),
                                          isolation_indices)
        isolate_other_data_tf = Subset(setup_seed.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                                              label_path=poisoned_set_label_path,
                                                              transforms=self.tf_compose_finetuning), other_indices)
        print("Isolated Poisoned Data Length:", len(isolate_poisoned_data_tf))
        print("Isolated Other Data Length:", len(isolate_other_data_tf))

        # create dataloaders
        isolate_poisoned_data_loader = DataLoader(isolate_poisoned_data_tf, batch_size=self.batch_size_unlearning,
                                                  shuffle=True, num_workers=4, pin_memory=True)
        isolate_other_data_loader = DataLoader(isolate_other_data_tf, batch_size=self.batch_size_finetuning,
                                               shuffle=True, num_workers=4, pin_memory=True)

        if self.finetuning_ascent_model:
            # this is to improve the clean accuracy of isolation model, you can skip this step
            print('----------- Finetuning isolation model --------------')
            for epoch in range(0, self.finetuning_epochs):
                self.learning_rate_finetuning(optimizer, epoch)
                self.train_step_finetuing(isolate_other_data_loader, model_ascent, optimizer, criterion, epoch + 1)
                # val_atk(self.args, model_ascent)
                test(model_ascent, test_loader=self.test_loader, poison_test=True,
                     poison_transform=self.poison_transform, num_classes=self.num_classes,
                     source_classes=self.source_classes)

            # save finetuned model
            self.save_checkpoint({
                'epoch': self.finetuning_epochs,
                'state_dict': model_ascent.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, self.finetuning_epochs, True, phase='finetuning')
        elif os.path.exists(os.path.join(self.folder_path, 'abl_%s_finetuning_epoch=%d_seed=%d.tar' % (
                supervisor.get_dir_core(self.args), self.finetuning_epochs, self.args.seed))):
            self.load_checkpoint(model=model_ascent,
                                 filepath=os.path.join(self.folder_path, 'abl_%s_finetuning_epoch=%d_seed=%d.tar' % (
                                     supervisor.get_dir_core(self.args), self.finetuning_epochs, self.args.seed)))
            print('Loaded ascent model (finetuning)!')

        print('----------- Model unlearning --------------')
        # freeze batchnorm runtime estimation
        for name, module in list(model_ascent.named_modules()):
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0

        for epoch in range(0, self.unlearning_epochs):
            self.learning_rate_unlearning(optimizer, epoch)

            # train stage
            if epoch == 0:
                # test firstly
                # val_atk(self.args, model_ascent)
                test(model_ascent, test_loader=self.test_loader, poison_test=True,
                     poison_transform=self.poison_transform, num_classes=self.num_classes,
                     source_classes=self.source_classes)
            self.train_step_unlearning(isolate_poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)

            # evaluate on testing set
            # val_atk(self.args, model_ascent)
            test(model_ascent, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform,
                 num_classes=self.num_classes, source_classes=self.source_classes)

        # save unlearned model
        self.save_checkpoint({
            'epoch': self.unlearning_epochs,
            'state_dict': model_ascent.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, self.unlearning_epochs, True, phase='unlearning')

        save_path = os.path.join(self.folder_path, "ABL_%s_seed=%d.pt" % (
            supervisor.get_dir_core(args, include_model_name=False),
            self.args.seed))
        torch.save(model_ascent.state_dict(), save_path)
        print("[Save] Unlearned model saved to %s" % save_path)

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch < self.isolation_epochs:
            lr = self.tuning_lr
        else:
            lr = self.tuning_lr * 0.1
        # print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def learning_rate_finetuning(self, optimizer, epoch):
        if epoch < 40:
            lr = self.lr_finetuning_init
        elif epoch < 60:
            lr = self.lr_finetuning_init * 0.1
        else:
            lr = self.lr_finetuning_init * 0.01
        # print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def learning_rate_unlearning(self, optimizer, epoch):
        if epoch < self.unlearning_epochs:
            lr = self.lr_unlearning
        else:
            lr = self.lr_unlearning * 0.2
        # print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, state, epoch, is_best, phase='isolation'):
        if is_best:
            filepath = os.path.join(self.folder_path, 'abl_%s_%s_epoch=%d_seed=%d.tar' % (
                supervisor.get_dir_core(self.args), phase, epoch, self.args.seed))
            torch.save(state, filepath)
        print('[info] Saved model and metainfo at \'%s\'' % filepath)

    def load_checkpoint(self, model, epoch=None, filepath=None):
        if epoch is None: epoch = self.isolation_epochs
        if filepath is None: filepath = os.path.join(self.folder_path, 'abl_%s_isolation_epoch=%d_seed=%d.tar' % (
            supervisor.get_dir_core(self.args), epoch, self.args.seed))
        print('Loading Model from {}'.format(filepath))
        checkpoint = torch.load(filepath, map_location='cpu')
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'])

        checkpoint_epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {}) ".format(filepath, checkpoint['epoch']))

        return model, checkpoint_epoch


# endregion ABL


if __name__ == '__main__':
    dataset = 'cifar10'
    args = config.Arguments()

    setup_seed.setup_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

    start_time = time.perf_counter()

    if args.defense == 'ABL':
        defense = ABL(
            args,
            isolation_epochs=15,
            isolation_ratio=0.001,
            # gradient_ascent_type='LGA',
            gradient_ascent_type='Flooding',
            gamma=0.01,
            flooding=0.3,
            do_isolate=True,
            finetuning_ascent_model=True,
            finetuning_epochs=60,
            unlearning_epochs=5,
            lr_unlearning=2e-2,
            do_unlearn=True,
        )
        defense.detect()

    end_time = time.perf_counter()
    print("Elapsed time: {:.2f}s".format(end_time - start_time))

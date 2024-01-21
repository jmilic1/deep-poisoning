import argparse
import os

import torch
from torch import nn
from torchvision import transforms

import config
from train_on_poisoned_set import get_poison_transform
from utils import supervisor, tools

if __name__ == '__main__':
    args = config.Arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

    batch_size = 128
    kwargs = {'num_workers': 4, 'pin_memory': True}

    data_transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])

    num_classes = 10
    arch = args.arch
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 200
    milestones = torch.tensor([100, 150])
    learning_rate = 0.1

    poison_set_dir = supervisor.get_poison_set_dir(args)
    model_path = supervisor.get_model_dir(args, cleanse=(args.cleanser is not None))

    model = arch(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = nn.DataParallel(model)
    model = model.cuda()
    print("Evaluating model '{}'...".format(model_path))

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, **kwargs)

    # Poison Transform for Testing
    poison_transform = get_poison_transform(poison_type=args.poison_type,
                                            target_class=args.target_class,
                                            trigger_transform=data_transform,
                                            alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                            trigger_name=args.trigger_name)

    if args.poison_type == 'TaCT':
        source_classes = [config.source_class]
    else:
        source_classes = None

    tools.test(model=model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform,
               num_classes=num_classes, source_classes=source_classes)

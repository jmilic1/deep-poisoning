import os
import sys

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from tqdm import tqdm

import config
from utils import resnet
from utils import supervisor, setup_seed
from utils.setup_seed import IMG_Dataset


def get_poison_transform(poison_type, target_class, trigger_transform=None,
                         alpha=0.2, trigger_name=None):
    # source class will be used for TaCT poison

    if trigger_name is None:
        trigger_name = config.trigger_default[poison_type]

    img_size = 32

    normalizer = transforms.Compose([
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    denormalizer = transforms.Compose([
        transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                             [1 / 0.247, 1 / 0.243, 1 / 0.261])
    ])

    if trigger_transform is None:
        trigger_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    if trigger_name != 'none':  # none for SIG
        trigger_path = os.path.join('./triggers', trigger_name)
        trigger = Image.open(trigger_path).convert("RGB")

        trigger_mask_path = os.path.join('./triggers', 'mask_%s' % trigger_name)

        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            temp_trans = transforms.ToTensor()
            trigger_map = temp_trans(trigger)
            trigger_mask = torch.logical_or(torch.logical_or(trigger_map[0] > 0, trigger_map[1] > 0),
                                            trigger_map[2] > 0).float()

        trigger = trigger_transform(trigger)
        trigger_mask = trigger_mask

    if poison_type == 'badnet':
        def transform(data, labels):
            data = data.clone()
            labels = labels.clone()

            _, dx, dy = trigger.shape

            # transform clean samples to poison samples
            posx = img_size - dx
            posy = img_size - dy
            labels[:] = target_class
            data[:, :, posx:, posy:] = trigger
            return data, labels

        return transform

    elif poison_type == 'adaptive_blend':
        def transform(data, labels):
            data, labels = data.clone(), labels.clone()
            data = data + alpha * (trigger - data)
            labels[:] = target_class

            return data, labels

        return transform

    elif poison_type == 'adaptive_patch':
        def transform(data, labels):
            img_size = 32
            target_class = 0
            test_trigger_names = ['phoenix_corner2_32.png', 'badnet_patch4_32.png']

            # triggers
            trigger_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            trigger_marks = []
            trigger_masks = []
            alphas = [1, 1]
            for i in range(len(test_trigger_names)):
                trigger_path = os.path.join('./triggers', test_trigger_names[i])
                trigger_mask_path = os.path.join('./triggers', 'mask_%s' % test_trigger_names[i])
                trigger = Image.open(trigger_path).convert("RGB")
                trigger = trigger_transform(trigger)
                if os.path.exists(
                        trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
                    trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                    trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
                else:  # by default, all black pixels are masked with 0's
                    trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                                                    trigger[2] > 0).float()

                trigger_marks.append(trigger)
                trigger_masks.append(trigger_mask)

            data, labels = data.clone(), labels.clone()

            data = denormalizer(data)
            for j in range(len(trigger_marks)):
                data = data + alphas[j] * trigger_masks[j] * (trigger_marks[j] - data)
            data = normalizer(data)
            labels[:] = target_class

            return data, labels

        return transform

    return poison_transform


def test(model, test_loader, poison_test=False, poison_transform=None, num_classes=10, source_classes=None):
    model.eval()
    clean_correct = 0
    poison_correct = 0
    tot = 0
    num_non_target_class = 0
    criterion = nn.CrossEntropyLoss()
    tot_loss = 0
    poison_acc = 0

    class_dist = np.zeros((num_classes))

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data, target
            clean_output = model(data)
            clean_pred = clean_output.argmax(dim=1)
            clean_correct += clean_pred.eq(target).sum().item()

            tot += len(target)
            this_batch_size = len(target)
            tot_loss += criterion(clean_output, target) * this_batch_size

            for bid in range(this_batch_size):
                if clean_pred[bid] == target[bid]:
                    class_dist[target[bid]] += 1

            if poison_test:
                clean_target = target
                data, target = poison_transform(data, target)

                poison_output = model(data)
                poison_pred = poison_output.argmax(dim=1, keepdim=True)

                target_class = target[0].item()
                for bid in range(this_batch_size):
                    if clean_target[bid] != target_class:
                        if source_classes is None:
                            num_non_target_class += 1
                            if poison_pred[bid] == target_class:
                                poison_correct += 1
                        else:  # for source-specific attack
                            if clean_target[bid] in source_classes:
                                num_non_target_class += 1
                                if poison_pred[bid] == target_class:
                                    poison_correct += 1

                poison_acc += poison_pred.eq((clean_target.view_as(poison_pred))).sum().item()

    print('Clean ACC: {}/{} = {:.6f}, Loss: {}'.format(
        clean_correct, tot,
        clean_correct / tot, tot_loss / tot
    ))
    if poison_test:
        print('ASR: %d/%d = %.6f' % (poison_correct, num_non_target_class, poison_correct / num_non_target_class))
        # print('Attack ACC: %d/%d = %.6f' % (poison_acc, tot, poison_acc/tot) )
    print('Class_Dist: ', class_dist)
    print("")

    if poison_test:
        return clean_correct / tot, poison_correct / num_non_target_class
    return clean_correct / tot, None


if __name__ == '__main__':
    log = False
    dataset = 'cifar10'
    poison_type = 'adaptive_blend'
    poison_rate = 0.003
    cover_rate = 0.003
    alpha = 0.2
    trigger = {'adaptive_blend': 'hellokitty_32.png',
               'badnet': 'badnet_patch.png',
               'adaptive_patch': [
                   'phoenix_corner_32.png',
                   'firefox_corner_32.png',
                   'badnet_patch4_32.png',
                   'trojan_square_32.png',
               ],
               }[poison_type]
    seed = 100

    setup_seed.setup_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % 0
    if log:
        out_path = 'logs'
        if not os.path.exists(out_path): os.mkdir(out_path)
        out_path = os.path.join(out_path, '%s_seed=%s' % (dataset, seed))
        if not os.path.exists(out_path): os.mkdir(out_path)
        out_path = os.path.join(out_path, 'base')
        if not os.path.exists(out_path): os.mkdir(out_path)
        out_path = os.path.join(out_path, '%s_%s.out' % (
            supervisor.get_dir_core(dataset, poison_type, poison_rate, cover_rate, alpha, trigger, seed,
                                    include_poison_seed=False), 'aug'))
        fout = open(out_path, 'w')
        ferr = open('/dev/null', 'a')
        sys.stdout = fout
        sys.stderr = ferr

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

    batch_size = 128

    num_classes = 10
    arch = resnet.resnet20
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 200
    milestones = torch.tensor([100, 150])
    learning_rate = 0.1

    kwargs = {'num_workers': 2, 'pin_memory': True}

    # Set Up Poisoned Set
    poison_set_dir = supervisor.get_poison_set_dir(dataset, {
        poison_type: supervisor.posionConfig(poison_rate, cover_rate, alpha, trigger)})[
        poison_type]

    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    poisoned_set = IMG_Dataset(data_dir=poisoned_set_img_dir,
                               label_path=poisoned_set_label_path,
                               transforms=data_transform_aug)

    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, **kwargs)

    poisoned_set_loader_no_shuffle = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=False, **kwargs)

    poison_indices = torch.tensor(torch.load(poison_indices_path))

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = IMG_Dataset(data_dir=test_set_img_dir,
                           label_path=test_set_label_path, transforms=data_transform)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=True, **kwargs)

    # Poison Transform for Testing
    poison_transform = get_poison_transform(poison_type=poison_type, target_class=0,
                                            trigger_transform=data_transform,
                                            alpha=alpha,
                                            trigger_name=trigger, dataset=dataset)

    # Train Code
    model = arch(num_classes=num_classes)
    milestones = milestones.tolist()
    model = nn.DataParallel(model)

    model_dir = supervisor.get_model_dir(dataset, poison_type, {
        poison_type: supervisor.posionConfig(poison_rate, cover_rate, alpha, trigger)}, seed)
    print(f"Will save to '{model_dir}'.")

    if os.path.exists(model_dir):  # exit if there is an already trained model
        print(f"Model '{model_dir}' already exists!")
        exit(0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    print(model_dir)

    if poison_type == 'TaCT':
        source_classes = [1]
    else:
        source_classes = None

    for epoch in range(1, epochs + 1):  # train backdoored base model
        # Train
        model.train()
        preds = []
        labels = []
        for data, target in tqdm(poisoned_set_loader):
            optimizer.zero_grad()
            data, target = data, target  # train set batch
            output = model(data)
            preds.append(output.argmax(dim=1))
            labels.append(target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        train_acc = (torch.eq(preds, labels).int().sum()) / preds.shape[0]
        print(
            '\n<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, Train Acc: {:.6f}, lr: {:.2f}'.format(epoch,
                                                                                                         loss.item(),
                                                                                                         train_acc,
                                                                                                         optimizer.param_groups[
                                                                                                             0]['lr']))
        scheduler.step()

        if epoch % 20 == 0:
            # Test
            test(model=model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform,
                 num_classes=num_classes, source_classes=source_classes)
            torch.save(model.module.state_dict(), model_dir)

    torch.save(model.module.state_dict(), model_dir)

import os
import sys

import numpy as np
import torch

import config
from torch import nn

"""
In our defensive setting, we assume a poisoned training set and a small clean 
set at hand, i.e. we train base model jointly with the poisoned set and 
the shifted set (constructed based on the small clean set).

On the other hand, we also prepare a clean test set (usually larger than the 
small clean set used for defense in our experiments). But note that, this set is 
not used for defense, it is used for debug and evaluation!

Below we implement tools to take use of the additional clean test set for debug & evaluation.
"""


class posionConfig():
    def __init__(self, poison_rate=0.2, cover_rate=0.0, alpha=0.0, trigger='none'):
        self.poison_rate = poison_rate
        self.cover_rate = cover_rate
        self.alpha = alpha
        self.trigger = trigger


def get_cleansed_set_indices_dir(args: config.Arguments):
    poison_set_dir = get_poison_set_dir(args)
    if args.cleanser == 'CT':  # confusion training
        return os.path.join(poison_set_dir, f'cleansed_set_indices_seed={args.seed}')
    else:
        return os.path.join(poison_set_dir, f'cleansed_set_indices_other_{args.cleanser}_seed={args.seed}')


def get_model_name(args: config.Arguments, cleanse=False):
    # `args.model_path` > `args.model` > by default 'full_base'
    if args.model_path is not None:
        model_name = args.model_path
    elif args.model is not None:
        model_name = args.model
    else:
        if args.no_aug:
            model_name = f'full_base_no_aug_seed={args.seed}.pt'
        else:
            model_name = f'full_base_aug_seed={args.seed}.pt'

        if cleanse and args.cleanser is not None:
            model_name = f"cleansed_{args.cleanser}_{model_name}"
    return model_name


def get_model_dir(args: config.Arguments, cleanse=False):
    if args.model_path is not None:
        return args.model_path

    return f"{get_poison_set_dir(args)}/{get_model_name(args, cleanse=cleanse)}"


def get_dir_core(args: config.Arguments, include_model_name=False, poison_seed=None):
    ratio = '%.3f' % args.poison_rate
    cover_rate = '%.3f' % args.cover_rate
    # ratio = '%.1f' % (args.poison_rate * 100) + '%'
    if args.poison_type == 'adaptive_blend':
        blend_alpha = '%.3f' % args.alpha
        dir_core = '%s_%s_%s_alpha=%s_cover=%s_trigger=%s' % (
            args.dataset, args.poison_type, ratio, blend_alpha, cover_rate, args.trigger_name)
    elif args.poison_type == 'adaptive_patch':
        dir_core = '%s_%s_%s_cover=%s' % (args.dataset, args.poison_type, ratio, cover_rate)
    else:
        dir_core = '%s_%s_%s' % (args.dataset, args.poison_type, ratio)

    if include_model_name:
        dir_core = f'{dir_core}_{get_model_name(args.model_path, args.model, args.no_aug, args.cleanser, args.seed)}'
    if poison_seed is not None:
        dir_core = f'{dir_core}_poison_seed={poison_seed}'
    return dir_core


def get_poison_set_dir(args: config.Arguments) -> str:
    ratio = '%3.f' % args.poison_rate
    blend_alpha = '%3.f' % args.alpha
    cover_rate = '%.3f' % args.cover_rate

    if args.poison_type == 'adaptive_blend':
        return 'poisoned_train_set/%s/%s_%s_alpha=%s_cover=%s_trigger=%s' % (
            args.dataset, args.poison_type, ratio, blend_alpha, cover_rate, args.trigger_name)
    if args.poison_type == 'adaptive_patch' or args.poison_type == 'TaCT':
        return 'poisoned_train_set/%s/%s_%s_cover=%s' % (args.dataset, args.poison_type, ratio, cover_rate)
    else:
        return 'poisoned_train_set/%s/%s_%s' % (args.dataset, args.poison_type, ratio)


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

            data, target = data.cuda(), target.cuda()
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
                data, target = poison_transform.transform(data, target)

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


def do_logging(args: config.Arguments):
    if args.log:
        out_path = 'logs'
        if not os.path.exists(out_path): os.mkdir(out_path)
        out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
        if not os.path.exists(out_path): os.mkdir(out_path)
        out_path = os.path.join(out_path, 'cleanse')
        if not os.path.exists(out_path): os.mkdir(out_path)
        out_path = os.path.join(out_path, '%s_%s.out' % (
            args.cleanser,
            get_dir_core(args)))
        fout = open(out_path, 'w')
        ferr = open('/dev/null', 'a')
        sys.stdout = fout
        sys.stderr = ferr

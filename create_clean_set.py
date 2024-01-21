import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import random
import ssl

import config
from utils import setup_seed

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    arguments = config.Arguments()
    # number of clean samples
    root_dir = "clean_set"
    setup_seed.setup_seed(arguments.seed)

    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    images = datasets.CIFAR10(os.path.join('./data', 'cifar10'), train=False,
                              download=True, transform=data_transform)

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    root_dir = os.path.join(root_dir, 'cifar10')
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    clean_split_dir = os.path.join(root_dir, "clean_split")
    if not os.path.exists(clean_split_dir):
        os.mkdir(clean_split_dir)

    clean_split_img_dir = os.path.join(clean_split_dir, 'data')
    if not os.path.exists(clean_split_img_dir):
        os.mkdir(clean_split_img_dir)

    test_split_dir = os.path.join(root_dir, 'test_split')
    if not os.path.exists(test_split_dir):
        os.mkdir(test_split_dir)

    test_split_img_dir = os.path.join(test_split_dir, 'data')
    if not os.path.exists(test_split_img_dir):
        os.mkdir(test_split_img_dir)

    num_img = len(images)
    id_set = list(range(0, num_img))
    random.shuffle(id_set)
    clean_split_indices = id_set[:arguments.num_clean]
    test_indices = id_set[arguments.num_clean:]

    # Construct Shift Set for Defensive Purpose
    clean_split_set = torch.utils.data.Subset(images, clean_split_indices)
    num = len(clean_split_set)

    clean_label_set = []

    for i in range(num):
        img, target_class = clean_split_set[i]
        img_file_name = '%d.png' % i
        img_file_path = os.path.join(clean_split_img_dir, img_file_name)
        save_image(img, img_file_path)
        print('[Generate Clean Split] Save %s' % img_file_path)
        clean_label_set.append(target_class)

    clean_label_set = torch.LongTensor(clean_label_set)
    clean_label_path = os.path.join(clean_split_dir, 'clean_labels')
    torch.save(clean_label_set, clean_label_path)
    print('[Generate Clean Split Set] Save %s' % clean_label_path)

    # Take the rest clean samples as the test set for debug & evaluation
    test_set = torch.utils.data.Subset(images, test_indices)
    num = len(test_set)
    label_set = []

    for i in range(num):
        img, gt = test_set[i]
        img_file_name = '%d.png' % i
        img_file_path = os.path.join(test_split_img_dir, img_file_name)
        save_image(img, img_file_path)
        print('[Generate Test Set] Save %s' % img_file_path)
        label_set.append(gt)

    label_set = torch.LongTensor(label_set)
    label_path = os.path.join(test_split_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Test Set] Save %s' % label_path)

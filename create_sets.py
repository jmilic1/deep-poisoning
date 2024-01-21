import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import random
from PIL import Image
import ssl

import config
from utils import supervisor
from math import sqrt

ssl._create_default_https_context = ssl._create_unverified_context


def get_trigger_mask(img_size, total_pieces, masked_pieces):
    div_num = sqrt(total_pieces)
    step = int(img_size // div_num)
    candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
    mask = torch.ones((img_size, img_size))
    for i in candidate_idx:
        x = int(i % div_num)  # column
        y = int(i // div_num)  # row
        mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 0
    return mask


def generate_badnet_poison_set(img_size, dataset, path, args: config.Arguments, trigger):
    # shape of the patch trigger
    _, dx, dy = trigger.shape

    # number of images
    num_img = len(dataset)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # poison for placing trigger pattern
    posx = img_size - dx
    posy = img_size - dy

    # random sampling
    id_set = list(range(0, num_img))
    random.shuffle(id_set)
    num_poison = int(num_img * args.poison_rate)
    poison_indices = id_set[:num_poison]
    poison_indices.sort()  # increasing order

    label_set = []
    pt = 0
    for i in range(num_img):
        img, target_of_image = dataset[i]

        # if pt < num_poison ??
        if pt < num_poison and poison_indices[pt] == i:
            target_of_image = args.target_class
            img[:, posx:, posy:] = trigger
            pt += 1

        img_file_name = '%d.png' % i
        img_file_path = os.path.join(path, img_file_name)
        save_image(img, img_file_path)
        # print('[Generate Poisoned Set] Save %s' % img_file_path)
        label_set.append(target_of_image)

    label_set = torch.LongTensor(label_set)

    return poison_indices, label_set


def generate_adaptive_blend_poison_set(img_size, dataset, args: config.Arguments, path, trigger):
    pieces = 16
    mask_rate = 0.5
    masked_pieces = round(mask_rate * pieces)

    # number of images
    num_img = len(dataset)

    torch.manual_seed(100)
    random.seed(100)

    # random sampling
    id_set = list(range(0, num_img))
    random.shuffle(id_set)
    num_poison = int(num_img * args.poison_rate)
    poison_indices = id_set[:num_poison]
    poison_indices.sort()  # increasing order

    num_cover = int(num_img * args.cover_rate)
    cover_indices = id_set[num_poison:num_poison + num_cover]  # use **non-overlapping** images to cover
    cover_indices.sort()

    label_set = []
    pt = 0
    ct = 0
    cnt = 0

    poison_id = []
    cover_id = []

    for i in range(num_img):
        img, target_of_image = dataset[i]

        # if ct < num_poison ??
        if ct < num_cover and cover_indices[ct] == i:
            cover_id.append(cnt)
            mask = get_trigger_mask(img_size, pieces, masked_pieces)
            img = img + args.alpha * mask * (trigger - img)
            ct += 1

        # if pt < num_poison ??
        if pt < num_poison and poison_indices[pt] == i:
            poison_id.append(cnt)
            target_of_image = args.target_class  # change the label to the target class
            mask = get_trigger_mask(img_size, pieces, masked_pieces)
            img = img + args.alpha * mask * (trigger - img)
            pt += 1

        img_file_name = '%d.png' % i
        img_file_path = os.path.join(path, img_file_name)
        save_image(img, img_file_path)
        print('[Generate Poisoned Set] Save %s' % img_file_path)
        label_set.append(target_of_image)
        cnt += 1

    label_set = torch.LongTensor(label_set)
    poison_indices = poison_id
    cover_indices = cover_id
    print("Poison indices:", poison_indices)
    print("Cover indices:", cover_indices)

    # demo
    img, gt = dataset[0]
    mask = get_trigger_mask(img_size, pieces, masked_pieces)
    img = img + args.alpha * mask * (trigger - img)
    save_image(img, os.path.join(path[:-4], 'demo.png'))

    return poison_indices, cover_indices, label_set


def generate_adaptive_patch_poison_set(dataset, path, trigger_names, alphas, args: config.Arguments):
    num_img = len(dataset)
    trigger_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    trigger_marks = []
    trigger_masks = []

    for i in range(len(trigger_names)):
        trigger_path = os.path.join('./triggers', trigger_names[i])
        trigger_mask_path = os.path.join('./triggers', 'mask_%s' % trigger_names[i])

        trigger = Image.open(trigger_path).convert("RGB")
        trigger = trigger_transform(trigger)

        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                                            trigger[2] > 0).float()

        trigger_marks.append(trigger)
        trigger_masks.append(trigger_mask)
        alphas.append(alphas[i])

        print(f"Trigger #{i}: {trigger_names[i]}")

        id_set = list(range(0, num_img))
        random.shuffle(id_set)
        num_poison = int(num_img * args.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort()  # increasing order

        num_cover = int(num_img * args.cover_rate)
        cover_indices = id_set[num_poison:num_poison + num_cover]  # use **non-overlapping** images to cover
        cover_indices.sort()

        label_set = []
        pt = 0
        ct = 0
        cnt = 0

        poison_id = []
        cover_id = []
        k = len(trigger_marks)

        for i in range(num_img):
            img, gt = dataset[i]

            # cover image
            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                for j in range(k):
                    if ct < (j + 1) * (num_cover / k):
                        img = img + alphas[j] * trigger_masks[j] * (trigger_marks[j] - img)
                        # img[j, :, :] = img[j, :, :] + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j][j, :, :] - img[j, :, :])
                        break
                ct += 1

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = args.target_class  # change the label to the target class
                for j in range(k):
                    if pt < (j + 1) * (num_poison / k):
                        img = img + alphas[j] * trigger_masks[j] * (trigger_marks[j] - img)
                        # img[j, :, :] = img[j, :, :] + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j][j, :, :] - img[j, :, :])
                        break
                pt += 1

            img_file_name = '%d.png' % cnt
            img_file_path = os.path.join(path, img_file_name)
            save_image(img, img_file_path)
            # print('[Generate Poisoned Set] Save %s' % img_file_path)

            label_set.append(gt)
            cnt += 1

        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        cover_indices = cover_id
        print("Poison indices:", poison_indices)
        print("Cover indices:", cover_indices)

        # demo
        img, gt = dataset[0]
        for j in range(k):
            img = img + alphas[j] * trigger_masks[j] * (trigger_marks[j] - img)
        save_image(img, os.path.join(path[:-4], 'demo.png'))

        return poison_indices, cover_indices, label_set


if __name__ == '__main__':
    args = config.Arguments()
    dataset = 'cifar10'
    data_dir = './data'

    print('[target class : %d]' % args.target_class)

    if not os.path.exists(os.path.join('poisoned_train_set', 'cifar10')):
        os.mkdir(os.path.join('poisoned_train_set', 'cifar10'))

    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                 download=True, transform=data_transform)
    img_size = 32
    num_classes = 10

    poison_set_dir = supervisor.get_poison_set_dir(args)
    poison_set_img_dir = os.path.join(poison_set_dir, 'data')

    if not os.path.exists(poison_set_dir):
        os.mkdir(poison_set_dir)
    if not os.path.exists(poison_set_img_dir):
        os.mkdir(poison_set_img_dir)

    trigger = None
    trigger_mask = None

    if args.trigger_name != '':
        trigger_path = os.path.join('./triggers', args.trigger_name)
        print('trigger: %s' % trigger_path)

        trigger = Image.open(trigger_path).convert("RGB")
        trigger = transforms.Compose([
            transforms.ToTensor()
        ])(trigger)

        trigger_mask_path = os.path.join('./triggers', 'mask_%s' % args.trigger_name)
        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            # print('trigger_mask_path:', trigger_mask_path)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            # print('No trigger mask found! By default masking all black pixels...')
            trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                                            trigger[2] > 0).float()

    poison_indices = None
    cover_indices = None
    label_set = None

    if args.poison_type == 'badnet':
        poison_indices, label_set = generate_badnet_poison_set(img_size=img_size, dataset=train_set, trigger=trigger,
                                                               path=poison_set_img_dir, args=args)
    if args.poison_type == 'adaptive_blend':
        poison_indices, cover_indices, label_set = generate_adaptive_blend_poison_set(img_size=img_size,
                                                                                      dataset=train_set,
                                                                                      args=args,
                                                                                      path=poison_set_img_dir,
                                                                                      trigger=trigger)
    if args.poison_type == 'adaptive_patch':
        poison_indices, cover_indices, label_set = generate_adaptive_patch_poison_set(dataset=train_set,
                                                                                      path=poison_set_img_dir,
                                                                                      trigger_names=[
                                                                                          'phoenix_corner_32.png',
                                                                                          'firefox_corner_32.png',
                                                                                          'badnet_patch4_32.png',
                                                                                          'trojan_square_32.png', ],
                                                                                      alphas=[
                                                                                          0.5,
                                                                                          0.2,
                                                                                          0.5,
                                                                                          0.3,
                                                                                      ],
                                                                                      args=args)
    if args.poison_type not in ['adaptive_blend', 'adaptive_patch', 'adaptive_k_way']:
        print('[Generate Poisoned Set] Save %d Images' % len(label_set))
    else:
        print('[Generate Poisoned Set] Save %d Images' % len(label_set))

        cover_indices_path = os.path.join(poison_set_dir, 'cover_indices')
        torch.save(cover_indices, cover_indices_path)
        print('[Generate Poisoned Set] Save %s' % cover_indices_path)

    label_path = os.path.join(poison_set_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Poisoned Set] Save %s' % label_path)

    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    torch.save(poison_indices, poison_indices_path)
    print('[Generate Poisoned Set] Save %s' % poison_indices_path)

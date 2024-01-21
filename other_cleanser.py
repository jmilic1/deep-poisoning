import argparse
import os
import sys
import random

from sklearn.metrics import silhouette_score

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

import config
from train_on_poisoned_set import get_poison_transform
from utils import supervisor, setup_seed, resnet


def spectral_signature_get_features(data_loader, model, num_classes):
    model.eval()
    class_indices = [[] for _ in range(num_classes)]
    feats = []

    with torch.no_grad():
        sid = 0
        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
            ins_data = ins_data.cuda()
            _, x_feats = model(ins_data, True)
            this_batch_size = len(ins_target)
            for bid in range(this_batch_size):
                feats.append(x_feats[bid].cpu().numpy())
                b_target = ins_target[bid].item()
                class_indices[b_target].append(sid + bid)
            sid += this_batch_size
    return feats, class_indices


def spectral_signature_cleanser(inspection_set, model, num_classes, poison_rate):
    """
        adapted from : https://github.com/hsouri/Sleeper-Agent/blob/master/forest/filtering_defenses.py
    """

    kwargs = {'num_workers': 4, 'pin_memory': True}
    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=128, shuffle=False, **kwargs)

    # Spectral Signature requires an expected poison ratio (we allow the oracle here as a baseline)
    num_poisons_expected = poison_rate * len(
        inspection_set) * 1.5  # allow removing additional 50% (following the original paper)

    feats, class_indices = spectral_signature_get_features(inspection_split_loader, model, num_classes)

    suspicious_indices = []

    for i in range(num_classes):

        if len(class_indices[i]) > 1:

            temp_feats = np.array([feats[temp_idx] for temp_idx in class_indices[i]])
            temp_feats = torch.FloatTensor(temp_feats)

            mean_feat = torch.mean(temp_feats, dim=0)
            temp_feats = temp_feats - mean_feat
            _, _, V = torch.svd(temp_feats, compute_uv=True, some=False)

            vec = V[:, 0]  # the top right singular vector is the first column of V
            vals = []
            for j in range(temp_feats.shape[0]):
                vals.append(torch.dot(temp_feats[j], vec).pow(2))

            k = min(int(num_poisons_expected), len(vals) // 2)
            # default assumption : at least a half of samples in each class is clean

            _, indices = torch.topk(torch.tensor(vals), k)
            for temp_index in indices:
                suspicious_indices.append(class_indices[i][temp_index])

    return suspicious_indices


# region AC

def activation_clustering_get_features(data_loader, model, num_classes):
    model.eval()
    class_indices = [[] for _ in range(num_classes)]
    feats = []

    with torch.no_grad():
        sid = 0
        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
            ins_data = ins_data.cuda()
            _, x_feats = model(ins_data, True)
            this_batch_size = len(ins_target)
            for bid in range(this_batch_size):
                feats.append(x_feats[bid].cpu().numpy())
                b_target = ins_target[bid].item()
                class_indices[b_target].append(sid + bid)
            sid += this_batch_size
    return feats, class_indices


def activation_clustering_cleanser(inspection_set, model, num_classes):
    """
        adapted from : https://github.com/hsouri/Sleeper-Agent/blob/master/forest/filtering_defenses.py
    """

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    kwargs = {'num_workers': 4, 'pin_memory': True}

    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=128, shuffle=False, **kwargs)

    suspicious_indices = []
    feats, class_indices = activation_clustering_get_features(inspection_split_loader, model, num_classes)

    for target_class in range(num_classes):
        if len(class_indices[target_class]) <= 1: continue  # no need to perform clustering...

        temp_feats = np.array([feats[temp_idx] for temp_idx in class_indices[target_class]])
        temp_feats = temp_feats - temp_feats.mean(axis=0)
        projector = PCA(n_components=10)

        projected_feats = projector.fit_transform(temp_feats)
        kmeans = KMeans(n_clusters=2, max_iter=2000).fit(projected_feats)

        # by default, take the smaller cluster as the poisoned cluster
        if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
            clean_label = 1
        else:
            clean_label = 0

        outliers = []
        for (bool, idx) in zip((kmeans.labels_ != clean_label).tolist(), list(range(len(kmeans.labels_)))):
            if bool:
                outliers.append(class_indices[target_class][idx])

        score = silhouette_score(projected_feats, kmeans.labels_)
        print('[class-%d] silhouette_score = %f' % (target_class, score))
        # if score > threshold:# and len(outliers) < len(kmeans.labels_) * 0.35:
        if len(outliers) < len(kmeans.labels_) * 0.35:  # if one of the two clusters is abnormally small
            print(f"Outlier Num in Class {target_class}:", len(outliers))
            suspicious_indices += outliers

    return suspicious_indices


# endregion AC

# region SCan
EPS = 1e-5


def calc_anomaly_index(a):
    ma = np.median(a)
    b = abs(a - ma)
    mm = np.median(b) * 1.4826
    index = b / mm
    return index


class SCAn:
    def __init__(self):
        self.lc_model = None
        self.gb_model = None

    def calc_final_score(self, lc_model=None):
        if lc_model is None:
            lc_model = self.lc_model
        sts = lc_model['sts']
        y = sts[:, 1]
        ai = calc_anomaly_index(y / np.max(y))
        return ai

    def build_global_model(self, reprs, labels, n_classes):
        N = reprs.shape[0]  # num_samples
        M = reprs.shape[1]  # len_features
        L = n_classes

        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        cnt_L = np.zeros(L)
        mean_f = np.zeros([L, M])
        for k in range(L):
            idx = (labels == k)
            cnt_L[k] = np.sum(idx)
            mean_f[k] = np.mean(X[idx], axis=0)

        u = np.zeros([N, M])
        e = np.zeros([N, M])
        for i in range(N):
            k = labels[i]
            u[i] = mean_f[k]  # class-mean
            e[i] = X[i] - u[i]  # sample-variantion
        Su = np.cov(np.transpose(u))
        Se = np.cov(np.transpose(e))

        # EM
        dist_Su = 1e5
        dist_Se = 1e5
        n_iters = 0
        while (dist_Su + dist_Se > EPS) and (n_iters < 100):
            n_iters += 1
            last_Su = Su
            last_Se = Se

            F = np.linalg.pinv(Se)
            SuF = np.matmul(Su, F)

            G_set = list()
            for k in range(L):
                G = -np.linalg.pinv(cnt_L[k] * Su + Se)
                G = np.matmul(G, SuF)
                G_set.append(G)

            u_m = np.zeros([L, M])
            e = np.zeros([N, M])
            u = np.zeros([N, M])

            for i in range(N):
                vec = X[i]
                k = labels[i]
                G = G_set[k]
                dd = np.matmul(np.matmul(Se, G), np.transpose(vec))
                u_m[k] = u_m[k] - np.transpose(dd)

            for i in range(N):
                vec = X[i]
                k = labels[i]
                e[i] = vec - u_m[k]
                u[i] = u_m[k]

            # max-step
            Su = np.cov(np.transpose(u))
            Se = np.cov(np.transpose(e))

            dif_Su = Su - last_Su
            dif_Se = Se - last_Se

            dist_Su = np.linalg.norm(dif_Su)
            dist_Se = np.linalg.norm(dif_Se)
            # print(dist_Su,dist_Se)

        gb_model = dict()
        gb_model['Su'] = Su
        gb_model['Se'] = Se
        gb_model['mean'] = mean_f
        self.gb_model = gb_model
        return gb_model

    def build_local_model(self, reprs, labels, gb_model, n_classes):
        Su = gb_model['Su']
        Se = gb_model['Se']

        F = np.linalg.pinv(Se)
        N = reprs.shape[0]
        M = reprs.shape[1]
        L = n_classes

        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        class_score = np.zeros([L, 3])
        u1 = np.zeros([L, M])
        u2 = np.zeros([L, M])
        split_rst = list()

        for k in range(L):
            selected_idx = (labels == k)
            cX = X[selected_idx]
            subg, i_u1, i_u2 = self.find_split(cX, F)
            # print("subg",subg)

            i_sc = self.calc_test(cX, Su, Se, F, subg, i_u1, i_u2)
            split_rst.append(subg)
            u1[k] = i_u1
            u2[k] = i_u2
            class_score[k] = [k, i_sc, np.sum(selected_idx)]

        lc_model = dict()
        lc_model['sts'] = class_score
        lc_model['mu1'] = u1
        lc_model['mu2'] = u2
        lc_model['subg'] = split_rst

        self.lc_model = lc_model
        return lc_model

    def find_split(self, X, F):
        N = X.shape[0]
        M = X.shape[1]
        subg = np.random.rand(N)

        if N == 1:
            subg[0] = 0
            return subg, X.copy(), X.copy()

        if np.sum(subg >= 0.5) == 0:
            subg[0] = 1
        if np.sum(subg < 0.5) == 0:
            subg[0] = 0
        last_z1 = -np.ones(N)

        # EM
        steps = 0
        while (np.linalg.norm(subg - last_z1) > EPS) and (np.linalg.norm((1 - subg) - last_z1) > EPS) and (steps < 100):
            steps += 1
            last_z1 = subg.copy()

            # max-step
            # calc u1 and u2
            idx1 = (subg >= 0.5)
            idx2 = (subg < 0.5)
            if (np.sum(idx1) == 0) or (np.sum(idx2) == 0):
                break
            if np.sum(idx1) == 1:
                u1 = X[idx1]
            else:
                u1 = np.mean(X[idx1], axis=0)
            if np.sum(idx2) == 1:
                u2 = X[idx2]
            else:
                u2 = np.mean(X[idx2], axis=0)

            bias = np.matmul(np.matmul(u1, F), np.transpose(u1)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
            e2 = u1 - u2  # (64,1)
            for i in range(N):
                e1 = X[i]
                delta = np.matmul(np.matmul(e1, F), np.transpose(e2))
                if bias - 2 * delta < 0:
                    subg[i] = 1
                else:
                    subg[i] = 0

        return subg, u1, u2

    def calc_test(self, X, Su, Se, F, subg, u1, u2):
        N = X.shape[0]
        M = X.shape[1]

        G = -np.linalg.pinv(N * Su + Se)
        mu = np.zeros([1, M])
        for i in range(N):
            vec = X[i]
            dd = np.matmul(np.matmul(Se, G), np.transpose(vec))
            mu = mu - dd

        b1 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u1, F), np.transpose(u1))
        b2 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
        n1 = np.sum(subg >= 0.5)
        n2 = N - n1
        sc = n1 * b1 + n2 * b2

        for i in range(N):
            e1 = X[i]
            if subg[i] >= 0.5:
                e2 = mu - u1
            else:
                e2 = mu - u2
            sc -= 2 * np.matmul(np.matmul(e1, F), np.transpose(e2))

        return sc / N


def scan_get_features(data_loader, model):
    class_indices = []
    feats = []

    model.eval()
    with torch.no_grad():
        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
            ins_data = ins_data.cuda()
            _, x_features = model(ins_data, True)

            this_batch_size = len(ins_target)
            for bid in range(this_batch_size):
                feats.append(x_features[bid].cpu().numpy())
                class_indices.append(ins_target[bid].cpu().numpy())

    return feats, class_indices


def scan_cleanser(inspection_set, clean_set, model, num_classes):
    kwargs = {'num_workers': 4, 'pin_memory': True}

    # main dataset we aim to cleanse
    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=128, shuffle=False, **kwargs)

    # a small clean batch for defensive purpose
    clean_set_loader = torch.utils.data.DataLoader(
        clean_set,
        batch_size=128, shuffle=True, **kwargs)

    feats_inspection, class_indices_inspection = scan_get_features(inspection_split_loader, model)
    feats_clean, class_indices_clean = scan_get_features(clean_set_loader, model)

    feats_inspection = np.array(feats_inspection)
    class_indices_inspection = np.array(class_indices_inspection)

    feats_clean = np.array(feats_clean)
    class_indices_clean = np.array(class_indices_clean)

    # For MobileNet-V2:
    # from sklearn.decomposition import PCA
    # projector = PCA(n_components=128)
    # feats_inspection = projector.fit_transform(feats_inspection)
    # feats_clean = projector.fit_transform(feats_clean)

    scan = SCAn()

    # fit the clean distribution with the small clean split at hand
    gb_model = scan.build_global_model(feats_clean, class_indices_clean, num_classes)

    size_inspection_set = len(feats_inspection)

    feats_all = np.concatenate([feats_inspection, feats_clean])
    class_indices_all = np.concatenate([class_indices_inspection, class_indices_clean])

    # use the global model to divide samples
    lc_model = scan.build_local_model(feats_all, class_indices_all, gb_model, num_classes)

    # statistic test for the existence of "two clusters"
    score = scan.calc_final_score(lc_model)
    threshold = np.e

    suspicious_indices = []

    for target_class in range(num_classes):

        print('[class-%d] outlier_score = %f' % (target_class, score[target_class]))

        if score[target_class] <= threshold: continue

        tar_label = (class_indices_all == target_class)
        all_label = np.arange(len(class_indices_all))
        tar = all_label[tar_label]

        cluster_0_indices = []
        cluster_1_indices = []

        cluster_0_clean = []
        cluster_1_clean = []

        for index, i in enumerate(lc_model['subg'][target_class]):
            if i == 1:
                if tar[index] > size_inspection_set:
                    cluster_1_clean.append(tar[index])
                else:
                    cluster_1_indices.append(tar[index])
            else:
                if tar[index] > size_inspection_set:
                    cluster_0_clean.append(tar[index])
                else:
                    cluster_0_indices.append(tar[index])

        # decide which cluster is the poison cluster, according to clean samples' distribution
        if len(cluster_0_clean) < len(cluster_1_clean):  # if most clean samples are in cluster 1
            suspicious_indices += cluster_0_indices
        else:
            suspicious_indices += cluster_1_indices

    return suspicious_indices


# endregion SCan

# region Strip
class STRIP:
    name: str = 'strip'

    def __init__(self, args: config.Arguments, inspection_set, clean_set, model, strip_alpha: float = 0.5, N: int = 64,
                 defense_fpr: float = 0.05):

        self.args = args

        self.strip_alpha: float = strip_alpha
        self.N: int = N
        self.defense_fpr = defense_fpr

        self.inspection_set = inspection_set
        self.clean_set = clean_set

        self.model = model

    def cleanse(self):

        # choose a decision boundary with the test set
        clean_entropy = []
        clean_set_loader = torch.utils.data.DataLoader(self.clean_set, batch_size=128, shuffle=False)
        for _input, _label in tqdm(clean_set_loader):
            _input, _label = _input.cuda(), _label.cuda()
            entropies = self.check(_input, _label, self.clean_set)
            for e in entropies:
                clean_entropy.append(e)
        clean_entropy = torch.FloatTensor(clean_entropy)

        clean_entropy, _ = clean_entropy.sort()
        print(len(clean_entropy))
        threshold_low = float(clean_entropy[int(self.defense_fpr * len(clean_entropy))])
        threshold_high = np.inf

        # now cleanse the inspection set with the chosen boundary
        inspection_set_loader = torch.utils.data.DataLoader(self.inspection_set, batch_size=128, shuffle=False)
        all_entropy = []
        for _input, _label in tqdm(inspection_set_loader):
            _input, _label = _input.cuda(), _label.cuda()
            entropies = self.check(_input, _label, self.clean_set)
            for e in entropies:
                all_entropy.append(e)
        all_entropy = torch.FloatTensor(all_entropy)

        suspicious_indices = torch.logical_or(all_entropy < threshold_low,
                                              all_entropy > threshold_high).nonzero().reshape(-1)
        return suspicious_indices

    def check(self, _input: torch.Tensor, _label: torch.Tensor, source_set) -> torch.Tensor:
        _list = []

        samples = list(range(len(source_set)))
        random.shuffle(samples)
        samples = samples[:self.N]

        for i in samples:
            X, Y = source_set[i]
            X, Y = X.cuda(), Y.cuda()
            _test = self.superimpose(_input, X)
            entropy = self.entropy(_test).cpu().detach()
            _list.append(entropy)
            # _class = self.model.get_class(_test)
        return torch.stack(_list).mean(0)

    def superimpose(self, _input1: torch.Tensor, _input2: torch.Tensor, alpha: float = None):
        if alpha is None:
            alpha = self.strip_alpha

        result = _input1 + alpha * _input2
        return result

    def entropy(self, _input: torch.Tensor) -> torch.Tensor:
        # p = self.model.get_prob(_input)
        p = torch.nn.Softmax(dim=1)(self.model(_input)) + 1e-8
        return (-p * p.log()).sum(1)


def strip_cleanser(inspection_set, clean_set, model, args: config.Arguments):
    """
        adapted from : https://github.com/hsouri/Sleeper-Agent/blob/master/forest/filtering_defenses.py
    """

    worker = STRIP(args, inspection_set, clean_set, model, strip_alpha=1.0, N=100, defense_fpr=0.1)
    suspicious_indices = worker.cleanse()

    return suspicious_indices


# endregion Strip

if __name__ == '__main__':
    dataset = 'cifar10'

    args = config.Arguments()

    setup_seed.setup_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

    save_path = supervisor.get_cleansed_set_indices_dir(args)
    arch = resnet.resnet20

    num_classes = 10
    if args.no_normalize:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])

    batch_size = 512

    poison_set_dir = supervisor.get_poison_set_dir(args)

    # poisoned set
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poisoned_set = setup_seed.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                          label_path=poisoned_set_label_path, transforms=data_transform)

    # small clean split at hand for defensive usage
    clean_set_dir = os.path.join('clean_set', dataset, 'clean_split')
    clean_set_img_dir = os.path.join(clean_set_dir, 'data')
    clean_set_label_path = os.path.join(clean_set_dir, 'clean_labels')
    clean_set = setup_seed.IMG_Dataset(data_dir=clean_set_img_dir,
                                       label_path=clean_set_label_path, transforms=data_transform)

    model_list = []
    alias_list = []

    path = supervisor.get_model_dir(args)
    if (args.model_path is not None) or (args.model is not None):
        model_list.append(path)
        alias_list.append('assigned')

    else:
        args.no_aug = True
        model_list.append(path)
        alias_list.append(supervisor.get_model_name(args))

        args.no_aug = False
        model_list.append(path)
        alias_list.append(supervisor.get_model_name(args))

    best_remain_indices = None
    best_recall = -999
    best_fpr = 999
    best_path = None

    for (vid, path) in enumerate(model_list):

        ckpt = torch.load(path)

        # base model for poison detection
        model = arch(num_classes=num_classes)
        model.load_state_dict(ckpt)
        model = nn.DataParallel(model)
        model = model.cuda()
        model.eval()

        # oracle knowledge of poison indices for evaluating detectors
        if args.poison_type != 'none':
            poison_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices'))

        if not os.path.exists(save_path):
            suspicious_indices = []
            if args.cleanser == "SS":
                if args.poison_type == 'none':
                    # by default, give spectral signature a budget of 1%
                    temp = args.poison_rate
                    args.poison_rate = 0.01

                suspicious_indices = spectral_signature_cleanser(poisoned_set, model, num_classes, args.poison_rate)

                if args.poison_type == 'none':
                    args.poison_rate = temp

            elif args.cleanser == "AC":
                suspicious_indices = activation_clustering_cleanser(poisoned_set, model, num_classes)
            elif args.cleanser == "SCAn":
                suspicious_indices = scan_cleanser(poisoned_set, clean_set, model, num_classes)
            elif args.cleanser == 'Strip':
                suspicious_indices = strip_cleanser(poisoned_set, clean_set, model, args)

            remain_indices = []
            for i in range(len(poisoned_set)):
                if i not in suspicious_indices:
                    remain_indices.append(i)
            remain_indices.sort()
        else:  # already exists, load from saved file
            print("Already cleansed!")
            remain_indices = torch.load(save_path)
            suspicious_indices = list(set(range(0, len(poisoned_set))) - set(remain_indices))
            suspicious_indices.sort()

        if True:
            if args.poison_type != 'none':
                true_positive = 0
                num_positive = len(poison_indices)
                false_positive = 0
                num_negative = len(poisoned_set) - num_positive

                suspicious_indices.sort()
                poison_indices.sort()

                pt = 0
                for pid in suspicious_indices:
                    while poison_indices[pt] < pid and pt + 1 < num_positive: pt += 1
                    if poison_indices[pt] == pid:
                        true_positive += 1
                    else:
                        false_positive += 1

                if not os.path.exists(save_path): print('<Overall Performance Evaluation with %s>' % path)
                tpr = true_positive / num_positive
                fpr = false_positive / num_negative
                if not os.path.exists(save_path): print(
                    'Elimination Rate = %d/%d = %f' % (true_positive, num_positive, tpr))
                if not os.path.exists(save_path): print(
                    'Sacrifice Rate = %d/%d = %f' % (false_positive, num_negative, fpr))

                if tpr > best_recall:
                    best_recall = tpr
                    best_remain_indices = remain_indices
                    best_fpr = fpr
                    best_path = path
                elif tpr == best_recall and fpr < best_fpr:
                    best_remain_indices = remain_indices
                    best_fpr = fpr
                    best_path = path


            else:

                print('<Test Cleanser on Clean Dataset with %s>' % path)
                false_positive = len(suspicious_indices)
                num_negative = len(poisoned_set)
                fpr = false_positive / num_negative
                print('Sacrifice Rate = %d/%d = %f' % (false_positive, num_negative, fpr))

                if fpr < best_fpr:
                    best_fpr = fpr
                    best_remain_indices = remain_indices
                    best_path = path

    if not os.path.exists(save_path):
        torch.save(best_remain_indices, save_path)
        print('[Save] %s' % save_path)
        print('best base model : %s' % best_path)

    if args.poison_type != 'none':
        num_positive = len(poison_indices)
        num_negative = len(poisoned_set) - num_positive
        print('Best Elimination Rate = %d/%d = %f' % (int(best_recall * num_positive), num_positive, best_recall))
        print('Best Sacrifice Rate = %d/%d = %f' % (int(best_fpr * num_negative), num_negative, best_fpr))
    else:
        num_negative = len(poisoned_set)
        print('Best Sacrifice Rate = %d/%d = %f' % (int(best_fpr * num_negative), num_negative, best_fpr))

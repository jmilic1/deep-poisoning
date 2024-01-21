import os
import sys

from utils import supervisor
from utils import resnet


class Arguments:
    def __init__(self):
        self.dataset = 'cifar10'
        self.num_clean = 2000
        self.seed = 100
        # can be ['badnet', 'adaptive_blend', 'adaptive_patch']
        self.poison_type = 'adaptive_blend'
        self.poison_rate = 0.003
        self.cover_rate = 0.003
        self.alpha = 0.2
        self.target_class = 0
        self.test_alpha = None
        self.no_aug = False
        self.model = None
        self.model_path = None
        self.no_normalize = False
        self.devices = 0
        self.arch = resnet.resnet20

        # choices=['ABL']
        self.defense = 'ABL'

        # choices ['SCAn', 'AC', 'SS', 'Strip']
        self.cleanser = 'Strip'
        self.log = False

        self.trigger_name = ''
        if self.poison_type == 'adaptive_blend':
            self.trigger_name = 'hellokitty_32.png'
        elif self.poison_type == 'badnet':
            self.trigger_name = 'badnet_patch.png'

    def do_logging(self, dataset='cifar10'):
        if self.log:
            out_path = 'logs'
            if not os.path.exists(out_path): os.mkdir(out_path)
            out_path = os.path.join(out_path, '%s_seed=%s' % (dataset, self.seed))
            if not os.path.exists(out_path): os.mkdir(out_path)
            out_path = os.path.join(out_path, 'cleanse')
            if not os.path.exists(out_path): os.mkdir(out_path)
            out_path = os.path.join(out_path, '%s_%s.out' % (
                self.cleanser,
                supervisor.get_dir_core(self)))
            fout = open(out_path, 'w')
            ferr = open('/dev/null', 'a')
            sys.stdout = fout
            sys.stderr = ferr

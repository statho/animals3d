'''
Generic testing utils -- Code borrowed and adapted from
https://github.com/nileshkulkarni/acsm/blob/master/acsm/nnutils/test_utils.py
'''

from __future__ import absolute_import, division, print_function
import os
import torch
from absl import flags


flags.DEFINE_string('name',           '', 'Experiment Name')
flags.DEFINE_integer('batch_size',    16, 'Size of minibatches')
flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')
flags.DEFINE_integer('iter_num',      -1, 'Number of training iterations')
flags.DEFINE_integer('tex_size',       6, 'Texture resolution per face')
flags.DEFINE_string('checkpoint_dir', 'acsm/cachedir/checkpoints', 'Directory where networks are saved')


class Tester():
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.device_count() else 'cpu')
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.device_count() else torch.Tensor
        self.save_dir = opts.checkpoint_dir
        if opts.name:
            self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_file = os.path.join(self.save_dir, 'opts_testing.log')
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))
        self.cam_location = [0, 0, -2.732]
        self.offset_z     = 5.0

    def dataparallel_model(self, epoch_label=-1):
        '''
        Checks whether DataParallel was used when the weights are saved.
        Models trained using DataParallel start with module.*
        '''
        save_name = f'{self.opts.category}.pth' if epoch_label==-1 else f'{epoch_label}.pth'
        save_path = os.path.join(self.save_dir, save_name)
        state_dict = torch.load(save_path)
        return list(state_dict.keys())[0].startswith('module.')

    def load_network(self, network, epoch_label=-1):
        save_name = f'{self.opts.category}.pth' if epoch_label==-1 else f'{epoch_label}.pth'
        save_path = os.path.join(self.save_dir, save_name)
        network.load_state_dict(torch.load(save_path))

    def init_dataset(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def test(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError
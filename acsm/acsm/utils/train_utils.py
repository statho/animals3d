'''
Generic training utils -- Code borrowed and adapted from
https://github.com/nileshkulkarni/acsm/blob/master/acsm/nnutils/train_utils.py
'''

from __future__ import absolute_import, division, print_function
import torch
import os
import time
from absl import flags
from tqdm import tqdm
from acsm.utils.visuals.visualizer import Visualizer


flags.DEFINE_string('name',              '', 'experiment Name')
flags.DEFINE_integer('num_iter',      70000, 'number of training iterations to train')
flags.DEFINE_integer('iter_save',      5000, 'save checkpoint every k iterations')
flags.DEFINE_integer('batch_size',       12, 'size of minibatches')
flags.DEFINE_integer('n_data_workers',    4, 'number of data loading workers')
flags.DEFINE_float('learning_rate',    1e-4, 'learning rate')
flags.DEFINE_float('beta1',             0.9, 'momentum term of adam')
flags.DEFINE_string('checkpoint_dir', 'acsm/cachedir/checkpoints', 'Directory where networks are saved')
flags.DEFINE_boolean('plot_scalars',   False, 'whether to plot scalars')
flags.DEFINE_integer('plot_freq',         20, 'scalar logging frequency')
flags.DEFINE_boolean('display_visuals',False, 'whether to display images')
flags.DEFINE_integer('display_freq',     200, 'visuals logging frequency')
flags.DEFINE_boolean('print_scalars',  False, 'whether to print scalars')
flags.DEFINE_integer('print_freq',       100, 'scalar logging frequency')
flags.DEFINE_integer('save_visual_count',  1, 'visuals save count')
flags.DEFINE_integer('tex_size',           6, 'Texture resolution per face')


class Trainer():
    def __init__(self, opts):
        self.opts   = opts
        self.device = torch.device('cuda' if torch.cuda.device_count() else 'cpu')
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.device_count() else torch.Tensor
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_file = os.path.join(self.save_dir, 'opts.log')
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))
        self.cam_location  = [0, 0, -2.732]
        self.offset_z      = 5.0
        self.sc_dict = {}
        self.use_articulations = False

    def dataparallel_model(self, epoch_label=-1):
        '''
        Checks whether DataParallel was used when the weights are saved.
        Models trained using DataParallel start with module.*
        '''
        save_name = f'{epoch_label}.pth'
        save_path = os.path.join(self.save_dir, save_name)
        state_dict = torch.load(save_path)
        return list(state_dict.keys())[0].startswith('module.')

    def save(self, network, epoch_label=-1):
        save_name = f'{str(epoch_label)}.pth'
        save_path = os.path.join(self.save_dir, save_name)
        torch.save(network.cpu().state_dict(), save_path)
        network.to(device=self.device)

    def init_dataset(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def define_criterion(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def forward(self):
        '''Should compute self.total_loss. To be implemented by the child class.'''
        raise NotImplementedError

    def get_current_visuals(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_scalars(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_points(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_training(self):
        opts = self.opts
        self.init_dataset()
        self.define_model()
        self.define_criterion()
        self.optimizer = torch.optim.Adam( self.model.parameters(), lr=opts.learning_rate, betas=(opts.beta1, 0.999) )

    def register_scalars(self, sc_dict, beta=0.99):
        '''
        Keeps a running smoothed average of some scalars.
        '''
        for k in sc_dict:
            if k not in self.sc_dict:
                self.sc_dict[k] = sc_dict[k]
            else:
                self.sc_dict[k] = beta * self.sc_dict[k] + (1 - beta) * sc_dict[k]

    def sample_data(self, loader):
        while True:
            for batch in loader:
                yield batch

    def train(self):
        opts = self.opts
        self.smoothed_total_loss = 0
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        start_time = time.time()

        loader = self.sample_data(self.dataloader)
        pbar = range(opts.num_iter)
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.01)
        self.real_iter = 0
        self.total_steps = 0

        ### Main Loop ###
        for idx in pbar:
            self.model.train()
            if idx > opts.num_iter:
                print("TRAINING IS DONE!")
                break

            batch = next(loader)
            self.set_input(batch)
            self.real_iter += 1
            self.total_steps += 1

            self.optimizer.zero_grad()
            self.forward()
            self.smoothed_total_loss = self.smoothed_total_loss * 0.99 + 0.01*self.total_loss.item()
            self.total_loss.backward()
            self.optimizer.step()

            # plot scalars
            if opts.plot_scalars and (self.total_steps % opts.plot_freq == 0):
                scalars = self.get_current_scalars()
                visualizer.plot_current_scalars(self.real_iter, scalars)

            # display visuals
            if opts.display_visuals and (self.total_steps % opts.display_freq == 0):
                visualizer.display_current_results(self.get_current_visuals())
                visualizer.plot_current_points(self.get_current_points())

            # print scalars
            if opts.print_scalars and (self.total_steps % opts.print_freq == 0):
                scalars = self.get_current_scalars()
                time_diff = time.time() - start_time
                visualizer.print_current_scalars(time_diff, self.real_iter, 0, scalars)

            # save checkpoint
            if self.total_steps % opts.iter_save == 0:
                print(f'=> saving model at iteration {int(self.total_steps / 1000)}K')
                self.save(network=self.model, epoch_label = int(self.total_steps / 1000))
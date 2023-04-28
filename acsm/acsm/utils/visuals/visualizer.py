'''
Code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
'''

import os
import time
import visdom
import numpy as np
import os.path as osp
from absl import flags
from acsm.utils.logger import Logger

server = 'http://localhost'
flags.DEFINE_string ('env_name',      'main', 'env name for experiments')
flags.DEFINE_integer('display_port',  16009, 'Display port')
flags.DEFINE_integer('display_winsize', 256, 'Display Size')
flags.DEFINE_integer('display_id',        1, 'Display Id')
flags.DEFINE_integer('display_single_pane_ncols', 0,
    'if positive, display all images in a single visdom web panel with certain number of images per row.'
)


class Visualizer():
    def __init__(self, opt):
        self.opts = opt
        self.name = opt.name
        self.env_name = self.name
        self.display_id = opt.display_id
        self.win_size = opt.display_winsize

        self.result_dir = osp.join('resss', opt.split, opt.env_name)

        if self.display_id > 0:
            print('Visdom Env Name {}'.format(self.env_name))
            self.vis = visdom.Visdom(server=server, port=opt.display_port, env=self.env_name)
            self.display_single_pane_ncols = opt.display_single_pane_ncols

        self.log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    def display_current_results(self, visuals):
        ''' visuals: dictionary of images to display or save '''
        if self.display_id > 0:
            if self.display_single_pane_ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
</style>""" % (w, h)
                ncols  = self.display_single_pane_ncols
                title  = self.name
                label_html     = ''
                label_html_row = ''
                nrows  = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx    = 0
                # for label, image_numpy in visuals.items():
                img_keys = visuals.keys()
                list.sort(img_keys)
                for label in img_keys:
                    image_numpy = visuals[label]
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1, padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2, opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image( image_numpy.transpose([2, 0, 1]), opts=dict(title=label), win=self.display_id + idx)
                    idx += 1

    def plot_current_scalars(self, iteration, scalars):
        ''' scalars: dictionary of scalar labels and values '''
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(scalars.keys())}
        self.plot_data['X'].append(iteration)
        self.plot_data['Y'].append( [scalars[k] for k in self.plot_data['legend']] )
        self.vis.line(
            X=np.stack( [np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1) ,
            Y=np.array(self.plot_data['Y']), opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'iteration',
                    'ylabel': 'loss'
                },
            win=self.display_id
        )

    def plot_current_points(self, points, disp_offset=10):
        ''' scatter plots '''
        idx = disp_offset
        for label, pts in points.items():
            self.vis.scatter(
                pts,
                opts=dict(title=label, markersize=1),
                win=self.display_id + idx
            )
            idx += 1

    def print_current_scalars(self, t, epoch, iteration, scalars):
        ''' scalars: same format as |scalars| of plot_current_scalars '''
        message = '(time : %0.3f, epoch: %d, iters: %d) ' % (t, epoch, iteration)
        for k, v in scalars.items():
            message += '%s: %.3f ' % (k, v)
        Logger.info(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
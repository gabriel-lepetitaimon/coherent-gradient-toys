import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.model import Model
from src.lines_generators.sample_generator import SampleGenerator
import time


def binary_dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    dice = 2*torch.sum(pred*target)/(torch.sum(pred+target)+eps)
    return 1-dice


class Experiment:
    def __init__(self, net: Model, gen_opt=None, loss=binary_dice_loss, gen2_opt=None) -> None:
        self.net = net.cuda()

        self.gen_opt = dict(
            profile=default_profile,
            orientation=default_orientation,
            length=default_length
        )
        if gen_opt is not None:
            self.gen_opt.update(gen_opt)
        self.samples_generator = SampleGenerator(5, 10, **self.gen_opt)

        if gen2_opt:
            def_secondary_gen_opt = dict(
                profile=default_profile,
                orientation=default_orientation,
                length=default_length
            )
            def_secondary_gen_opt.update(gen2_opt)
            self.secondary_generator = SampleGenerator(5, 10, **def_secondary_gen_opt)
            self.secondary_gen_mix = 0.5 if 'mix' not in gen2_opt else gen2_opt['mix']
        else:
            self.secondary_generator = None
            self.secondary_gen_mix = 0

        self.noise_std = .4
        self.noise_avg = .5

        self.loss = loss
        self.lr = 2e-2
        self.batchsize = 16
        self.early_stopping_delay = 15
        self.early_stopping_delta = .5e-3
        self.sample_height = 256

        self.train_iterations = 0
        self.last_chkpt = None
        self.training_logs = pd.DataFrame(columns=['iterations', 'loss'])

    def train(self, verbose=False):
        # Setup
        self.net.train()
        self.net.cuda()
        optimizer = torch.optim.Adam(self.net.parameters(), self.lr)

        best_loss = 1e12
        self.last_chkpt = self.net.state_dict()
        loss_improve_delay = 0

        start_t = time.time()
        if verbose:
            print(f' --- STARTING TRAINING ---')

        while loss_improve_delay < self.early_stopping_delay:
            x, y = self.generate_samples()
            optimizer.zero_grad()
            y_pred = self.net(x)
            loss = self.loss(y_pred, y)
            loss.backward()
            optimizer.step()

            loss = float(loss.cpu())

            self.train_iterations += 1
            if self.train_iterations % 10 == 0:
                if verbose:
                    print(f'[{self.train_iterations}] loss={loss:.3f}')
                self.training_logs = pd.concat([self.training_logs,
                    pd.DataFrame({'iterations': [self.train_iterations], 'loss': [loss],})
                ], ignore_index=True)

            if loss < best_loss:
                self.last_chkpt = self.net.state_dict()
                if best_loss - loss >= self.early_stopping_delta:
                    loss_improve_delay = 0
                    best_loss = loss
                else:
                    loss_improve_delay += 1
            else:
                loss_improve_delay += 1

        self.net.load_state_dict(self.last_chkpt)
        self.last_chkpt = None

        if verbose:
            print(f' --- TRAINING END in {int(time.time()-start_t):d}s ---')

    def generate_samples(self, b=None):
        if b is None:
            b = self.batchsize
        x, y, lines = self.samples_generator.generate(b=b, h=self.sample_height,
                                                      y_width=3, subsample=2, device='cuda')

        if self.secondary_generator:
            x2, _, _ = self.secondary_generator.generate(b=b, h=self.sample_height,
                                                         y_width=3, subsample=2, device='cuda')
            x = x2 * self.secondary_gen_mix + x * (1 - self.secondary_gen_mix)

        x = x[:, None].float()
        y = y[:, None].float()
        x_noise = torch.randn_like(x)*self.noise_std + self.noise_avg

        return x+x_noise, y

    def plot_test_samples(self):
        from matplotlib import pyplot as plt
        if self.last_chkpt is not None:
            self.net.load_state_dict(self.last_chkpt)
            self.last_chkpt = None

        x, y = self.generate_samples(b=6)
        y_sig = torch.sigmoid(self.net(x)).detach().cpu()

        x = x.cpu()
        y = y.cpu()

        fig, axs = plt.subplots(4, 6)
        x_plots, y_plots, y_sig_plots, y_pred_plots = axs
        for i in range(len(x_plots)):
            x_plots[i].imshow(x[i, 0])
            y_plots[i].imshow(y[i, 0])
            y_sig_plots[i].imshow(y_sig[i, 0])
            y_pred_plots[i].imshow(y_sig[i, 0]>=.4)
        for _axs in axs:
            for ax in _axs:
                ax.axis('off')
        fig.set_size_inches(18, 12)
        return fig


def default_profile(d):
    return torch.clip(1-torch.abs(d), 0)


def default_orientation(shape):
    return torch.rand(*shape)*(np.pi*2)


def default_length(shape):
    return torch.rand(*shape)*.2+.25
import torch
from torch import nn
from .clip_pad import cat_crop
from .convbn import ConvBN
from .model import Model


class ResNet(Model):
    def __init__(self, n_in, n_out=1, nfeatures_base=64, kernel=3, resblock_depth=2, resblock_count=3, nscale=4,
                 padding='same', p_dropout=0, batchnorm=True, downsampling='maxpooling', **kwargs):
        """
        :param n_in: Number of input features (or channels).
        :param n_out: Number of output features (or classes).
        :param nfeatures_base: Number of features of each convolution block for the first stage of the UNet. The number of features doubles at every stage.
        :param kernel: Height and Width of the convolution kernels
        :param resblock_depth: Number of convolution block per residual block.
        :param resblock_count: Number of residual block per downsampling stage.
        :param nscale: Number of downsampling stage.
        :param padding: Padding configuration [One of 'same' or 'auto'; 'true' or 'valid'; 'full'; number of padded pixel].
        :param p_dropout: Dropout probability during training.
        :param batchnorm: Adds batchnorm layers before each convolution layer.
        :param downsampling:
            - maxpooling: Maxpooling Layer.
            - averagepooling: Average Pooling.
            - conv: Stride on the last convolution.
        """
        super().__init__(n_in=n_in, n_out=n_out, nfeatures_base=nfeatures_base, resblock_depth=resblock_depth, resblock_count=resblock_count, nscale=nscale,
                         kernel=kernel, padding=padding, p_dropout=p_dropout, batchnorm=batchnorm,
                         downsampling=downsampling, **kwargs)

        # Down
        self.first_conv = self.setup_convbn(n_in, nfeatures_base)

        self.conv_stacks = []
        for i in range(nscale):
            nf = nfeatures_base * (2**i)
            resblocks = []
            for j in range(resblock_count):
                resblocks += [[self.setup_convbn(nf, nf, f'conv-{i}-{j}-{k}') for k in range(resblock_depth)]]
            if i+1<nscale and downsampling == 'conv':
                resblocks[-1][-1].stride = 2
            self.conv_stacks += [resblocks]

        # End
        self.final_conv = self.setup_convtranspose(nf, n_out, stride=2**(nscale-1))

        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else identity
        if downsampling == 'maxpooling':
            self.downsample = torch.nn.MaxPool2d(2)
        elif downsampling == 'averagepooling':
            self.downsample = torch.nn.AvgPool2d(2)
        elif downsampling == 'conv':
            self.downsample = identity
        else:
            raise ValueError(f'downsampling must be one of: "maxpooling", "averagepooling", "conv". '
                             f'Provided: {downsampling}.')
        self.res_downsample = torch.nn.AvgPool2d(2)

    def setup_convbn(self, n_in: int, n_out: int, name: str =None):
        conv = ConvBN(self.kernel, n_in, n_out, relu=True, bn=self.batchnorm, padding=self.padding)
        if name is not None:
            self.add_module(name, conv)
        return conv

    def setup_convtranspose(self, n_in, n_out, stride=2):
        return torch.nn.ConvTranspose2d(n_in, n_out, kernel_size=(stride, stride), stride=(stride, stride))

    def forward(self, x):
        """
        Args:
            x: The input tensor.

        Shape:
            input: (b, n_in, h, w)
            return: (b, n_out, ~h, ~w)

        Returns: The prediction of the network (without the sigmoid).

        """
        for i in range(self.nscale):
            for j in range(self.resblock_count):
                x_res = x
                if j == self.resblock_count-1 and self.downsampling=='conv':
                    x_res = self.res_downsample(x_res)
                x = self.reduce_stack(self.conv_stacks[i][j], x) + x_res
            x = self.dropout(x)
            x = self.downsample(x)

        return self.final_conv(x)

    def reduce_stack(self, conv_stack, x, **kwargs):
        from functools import reduce

        def conv(X, conv_mod):
            return conv_mod(X, **kwargs)
        return reduce(conv, conv_stack, x)

    @property
    def p_dropout(self):
        return self.dropout.p

    @p_dropout.setter
    def p_dropout(self, p):
        self.dropout.p = p


def identity(x):
    return x

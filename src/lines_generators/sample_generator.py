from .lines import Lines
import numpy as np
import torch


class SampleGenerator:
    def __init__(self, nlines, width, profile, orientation, length, subsample=1):
        self.nlines = nlines
        self.width = width
        self.profile = profile
        self.orientation = orientation
        self.length = length

    def generate(self, b=8, h=512, y_width=3, device='cuda', subsample=1):
        with torch.no_grad():
            nlines = self.nlines() if callable(self.nlines) else self.nlines

            shape = (nlines, b)
            orientation = self.orientation(shape) if callable(self.orientation) else self.orientation
            if isinstance(orientation, (int, float)):
                orientation = torch.Tensor([orientation])
                while orientation.ndim < 2:
                    orientation = orientation[None]
            elif isinstance(orientation, tuple):
                orientation = torch.rand(shape, device=device)*(orientation[1]-orientation[0]) + orientation[0]
            orientation = orientation.to(device)
            
            length = self.length(shape) if callable(self.length) else self.length
            if (isinstance(length, float) and length <= 1) or (isinstance(length, torch.Tensor) and (length<=1).all()):
                length *= h
            if isinstance(length, int):
                length = torch.Tensor([length])
                while length.ndim < 2:
                    length = length[None]
            length = length.to(device)

            dx, dy = torch.cos(orientation)*length/2, torch.sin(orientation)*length/2
            x = torch.rand(*shape, device=device)*(h-torch.abs(dx)*2)+torch.abs(dx)
            y = torch.rand(*shape, device=device)*(h-torch.abs(dx)*2)+torch.abs(dy)
            lines = Lines(a=(x+dx, y+dy), b=(x-dx, y-dy))

            dist_map = lines._distance_field(h, h, scale=subsample, roundtip=False, device=device)
            d_sign, d_abs = torch.sign(dist_map), torch.abs(dist_map)
            y = torch.amin(d_abs, dim=0)
            if subsample != 1:
                from torch.nn.functional import avg_pool2d
                y = avg_pool2d(y[None], subsample)[0]
            y =  y <= .5*y_width
            
            d_abs = d_abs/(self.width(shape) if callable(self.width) else self.width)
            d_arg = torch.argmin(d_abs, dim=0, keepdim=True)
            d_abs = torch.gather(d_abs, 0, d_arg)
            d_sign = torch.gather(d_sign, 0, d_arg)
            d = d_abs*d_sign
            x = Lines.dist_to_line(d[0], subsample=subsample, profile=self.profile)

        return x, y, lines

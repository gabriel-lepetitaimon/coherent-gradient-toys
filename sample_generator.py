from lines import Line
import numpy as np
import torch


class SampleGenerator:
    def __init__(self, nlines, width, profile, orientation, length):
        self.nlines = nlines
        self.width = width
        self.profile = profile
        self.orientation = orientation
        self.length = length

    def generate(self, b=8, h=512, y_width=3, device='cuda'):
        nlines = self.nlines() if callable(self.nlines) else self.nlines
        lines = []
        for i in range(b):
            blines = []
            for j in range(nlines):
                orientation = self.orientation() if callable(self.orientation) else self.orientation
                length = self.length() if callable(self.length) else self.length
                dx, dy = np.cos(orientation)*length/2, np.sin(orientation)*length/2
                x = np.random.rand()*(h-np.abs(dx)*2)+np.abs(dx)
                y = np.random.rand()*(h-np.abs(dx)*2)+np.abs(dy)
                blines += [Line(a=(x+dx, y+dy), b=(x-dx, y-dy))]
            lines += [blines]

        x = torch.zeros(b, 1, h, h, device=device)
        y = torch.zeros(b, h, h, device=device, dtype=torch.bool)

        for b, blines in enumerate(lines):
            for line in blines:
                profile = self.profile()
                width = self.width() if callable(self.width) else self.width
                x[b, 0] += line.draw_line(h, h, profile=profile, width=width, device=device, subsample=4)
                y[b] |= line.draw_line(h, h, device=device, width=y_width)

        return x, y, lines

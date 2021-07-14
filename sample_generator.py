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

    def generate(self, b=8, h=512, device='cuda'):
        nlines = self.nlines() if callable(self.nlines) else self.nlines
        lines = []
        for i in range(b):
            blines = []
            for j in range(nlines):
                width = self.width() if callable(self.width) else self.width
                orientation = self.orientation() if callable(self.orientation) else self.orientation
                length = self.length() if callable(self.length) else self.length
                profile = self.profile()
                dx, dy = np.cos(orientation)*length/2, np.sin(orientation)*length/2
                x = np.random.rand()*(h-np.abs(dx)*2)+np.abs(dx)
                y = np.random.rand()*(h-np.abs(dx)*2)+np.abs(dy)
                blines += [Line(a=(x+dx, y+dy), b=(x-dx, y-dy), profile=profile, width=width)]
            lines += [blines]

        x = torch.zeros(b, 1, h, h, device=device)
        y = torch.zeros(b, h, h, device=device, dtype=torch.bool)

        for b, blines in enumerate(lines):
            for line in blines:
                x[b, 0] += line.draw_line(h, h, device=device)
                y[b] |= line.draw_squeleton(h, h, device=device)

        return x, y, lines

import torch


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        if item == 0 or item == 'x':
            return self.x
        elif item == 1 or item == 'y':
            return self.y
        else:
            raise IndexError()

    def __setitem__(self, item, value):
        if item == 0 or item == 'x':
            self.x = value
        elif item == 1 or item == 'y':
            self.y = value
        else:
            raise IndexError()

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            self.x *= other
            self.y *= other
        elif isinstance(other, Point):
            self.x *= other.x
            self.y *= other.y
        else:
            raise TypeError()

    def __add__(self, other):
        if isinstance(other, (int, float)):
            self.x += other
            self.y += other
        elif isinstance(other, Point):
            self.x += other.x
            self.y += other.y
        else:
            raise TypeError()


class PointAttr:
    def __init__(self, x=0, y=0):
        self.point = Point(x, y)

    def __set__(self, obj, point):
        if isinstance(point, (tuple, list)):
            if len(point) != 2:
                raise AttributeError(f'2D Point should be tuple or list of size 2 not {len(point)}.\n '
                                     f'(Provided point: {point})')
            self.point.x, self.point.y = point
        elif isinstance(point, Point):
            self.point.x = point.x
            self.point.y = point.y

    def __get__(self, obj, objtype=None):
        return self.point


class Line:
    A = PointAttr()
    B = PointAttr()

    def __init__(self, a, b, profile, width):
        self.A = a
        self.B = b
        self.profile = profile
        self.width = width

    def _distance_field(self, h, w, scale=1):
        a = self.A * scale
        b = self.B * scale

    def draw_profile(self, h, w, subsample=1):
        d = self._distance_field(h*subsample, w*subsample, subsample) * self.width * subsample
        p = self.profile(d)
        if subsample != 1:
            from torch.nn.functional import avg_pool2d
            p = avg_pool2d(p, subsample)
        return p

    def draw_squeleton(self, h, w):
        pass
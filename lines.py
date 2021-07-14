import math
import torch
import numpy as np


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

    def __repr__(self):
        return f'(x={self.x:.2f}, y={self.y:.2f})'

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Point(self.x*other, self.y*other)
        elif isinstance(other, Point):
            return Point(self.x*other.x, self.y*other.y)
        else:
            raise TypeError()

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Point(self.x/other, self.y/other)
        elif isinstance(other, Point):
            return Point(self.x/other.x, self.y/other.y)
        else:
            raise TypeError()

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Point(other/self.x, other/self.y)
        elif isinstance(other, Point):
            return Point(other.x/self.x, other.y/self.y)
        else:
            raise TypeError()

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Point(self.x+other, self.y+other)
        elif isinstance(other, Point):
            return Point(self.x+other.x, self.y+other.y)
        else:
            raise TypeError()

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Point(self.x-other, self.y-other)
        elif isinstance(other, Point):
            return Point(self.x-other.x, self.y-other.y)
        else:
            raise TypeError()

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Point(other-self.x, other-self.y)
        elif isinstance(other, Point):
            return Point(other.x-self.x, other.y-self.y)
        else:
            raise TypeError()

    def vec_prod(self, other):
        return self.x*other.y-self.y*other.x

    def dot(self, other):
        x, y = np.dot(self, other)
        return Point(x, y)

    def norm(self):
        return np.linalg.norm(self)

    def unitary(self):
        return self / self.norm()

    def torch(self):
        return torch.Tensor(self)


class PointAttr:
    def __init__(self, x=0, y=0):
        self.default = Point(x, y)
        self.name = 'unknown'

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, obj, point):
        if isinstance(point, (tuple, list)):
            if len(point) != 2:
                raise AttributeError(f'2D Point should be tuple or list of size 2 not {len(point)}.\n '
                                     f'(Provided point: {point})')
            if hasattr(obj, '_'+self.name):
                p = getattr(obj, '_'+self.name)
                p.x, p.y = point
            else:
                setattr(obj, '_'+self.name, Point(*point))

        elif isinstance(point, Point):
            if hasattr(obj, '_'+self.name):
                p = getattr(obj, '_'+self.name)
                if hasattr(obj, '_'+self.name):
                    p = getattr(obj, '_'+self.name)
                p.x, p.y = point
            else:
                setattr(obj, '_'+self.name, Point(point.x, point.y))

    def __get__(self, obj, objtype=None):
        if not hasattr(obj, '_'+self.name):
            return self.default
        else:
            return getattr(obj, '_'+self.name)

    def _point(self, obj):
        return getattr(obj, '_'+self.name)


class Line:
    A = PointAttr()
    B = PointAttr()

    def __init__(self, a, b, profile, width):
        self.A = a
        self.B = b
        self.profile = profile
        self.width = width

    def __repr__(self):
        return f'Line: {self.A} - {self.B}'

    def _distance_field(self, h, w, scale=1, roundtip=False, device=None):
        a = self.A
        b = self.B
        u = (b-a).unitary()
        offset = (1-1/scale)/2
        x, y = torch.meshgrid(torch.arange(-offset, h-offset, 1/scale, device=device),
                              torch.arange(-offset, w-offset, 1/scale, device=device))
        d = u.x*(y-a.y) - u.y*(x-a.x)

        a_halfplane = (x-a.x)*u.x + (y-a.y)*u.y < 0
        b_halfplane = (x-b.x)*u.x + (y-b.y)*u.y > 0

        if roundtip:
            if not isinstance(roundtip, str):
                d[a_halfplane] = torch.sqrt(torch.square(x[a_halfplane]-a.x)+torch.square(y[a_halfplane]-a.y))
                d[b_halfplane] = torch.sqrt(torch.square(x[b_halfplane]-b.x)+torch.square(y[b_halfplane]-b.y))
            elif roundtip.lower() == 'a':
                d[a_halfplane] = torch.sqrt(torch.square(x[a_halfplane]-a.x)+torch.square(y[a_halfplane]-a.y))
                d[b_halfplane] = np.nan
            elif roundtip.lower() == 'b':
                d[a_halfplane] = np.nan
                d[b_halfplane] = torch.sqrt(torch.square(x[b_halfplane]-b.x)+torch.square(y[b_halfplane]-b.y))
            else:
                roundtip = False
        if not roundtip:
            halfplanes = a_halfplane | b_halfplane
            d[halfplanes] = np.nan
        return d

    def draw_line(self, h, w, subsample=1, roundtip=False, device=None):
        d = self._distance_field(h, w, subsample, roundtip, device=device) / self.width
        p = self.profile(d)
        p[p.isnan()] = 0
        if subsample != 1:
            from torch.nn.functional import avg_pool2d
            p = avg_pool2d(p[None], subsample)[0]
        return p

    def draw_squeleton(self, h, w, device=None):
        a, b = self.A, self.B

        if a.x == b.x:
            x = math.floor(a.x)
            y1 = math.floor(min(a.y, b.y))
            y2 = math.ceil(max(a.y, b.y))
            skeleton = torch.zeros((h,w), dtype=torch.bool, device=device)
            skeleton[x, y1:y2+1] = True
            return skeleton
        else:
            slope = (a.y-b.y) / (a.x-b.x)
            offset = (b.y*a.x - a.y*b.x) / (a.x-b.x)
            u = (b-a).unitary()
            x, y = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))

            y1 = slope*x+offset
            y2 = slope*(x+1)+offset
            skeleton = (((y <= y1) & (y1 < y+1)) | ((y <= y2) & (y2 < y+1)))
            if slope != 0:
                x1 = (y-offset)/slope
                skeleton |= (x <= x1) & (x1 < x+1)

            domain = ((x-a.x)*u.x + (y-a.y)*u.y >= 0) & ((x-b.x)*u.x + (y-b.y)*u.y <= 0)
            return skeleton & domain


class QuadBezierLine(Line):
    C = PointAttr()

    def __init__(self, a, b, c, profile, width):
        super(QuadBezierLine, self).__init__(a, b, profile, width)
        self.C = c

    def _distance_field(self, h, w, scale=1, roundtip=False):
        a = self.A
        b = self.B
        u = (b-a).unitary()
        offset = (1-1/scale)/2
        x, y = torch.meshgrid(torch.arange(-offset, h-offset, 1/scale),
                              torch.arange(-offset, w-offset, 1/scale))
        d = u.x*(y-a.y) - u.y*(x-a.x)

        a_halfplane = (x-a.x)*u.x + (y-a.y)*u.y < 0
        b_halfplane = (x-b.x)*u.x + (y-b.y)*u.y > 0

        if roundtip:
            if not isinstance(roundtip, str):
                d[a_halfplane] = torch.sqrt(torch.square(x[a_halfplane]-a.x)+torch.square(y[a_halfplane]-a.y))
                d[b_halfplane] = torch.sqrt(torch.square(x[b_halfplane]-b.x)+torch.square(y[b_halfplane]-b.y))
            elif roundtip.lower() == 'a':
                d[a_halfplane] = torch.sqrt(torch.square(x[a_halfplane]-a.x)+torch.square(y[a_halfplane]-a.y))
                d[b_halfplane] = np.nan
            elif roundtip.lower() == 'b':
                d[a_halfplane] = np.nan
                d[b_halfplane] = torch.sqrt(torch.square(x[b_halfplane]-b.x)+torch.square(y[b_halfplane]-b.y))
            else:
                roundtip = False
        if not roundtip:
            halfplanes = a_halfplane | b_halfplane
            d[halfplanes] = np.nan
        return d

    def draw_squeleton(self, h, w):
        a, b = self.A, self.B

        if a.x == b.x:
            x = math.floor(a.x)
            y1 = math.floor(min(a.y, b.y))
            y2 = math.ceil(max(a.y, b.y))
            skeleton = torch.zeros((h,w), dtype=torch.bool)
            skeleton[x, y1:y2+1] = True
            return skeleton
        else:
            slope = (a.y-b.y) / (a.x-b.x)
            offset = (b.y*a.x - a.y*b.x) / (a.x-b.x)
            u = (b-a).unitary()
            x, y = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))

            y1 = slope*x+offset
            y2 = slope*(x+1)+offset
            skeleton = (((y <= y1) & (y1 < y+1)) | ((y <= y2) & (y2 < y+1)))
            if slope != 0:
                x1 = (y-offset)/slope
                skeleton |= (x <= x1) & (x1 < x+1)

            domain = ((x-a.x)*u.x + (y-a.y)*u.y >= 0) & ((x-b.x)*u.x + (y-b.y)*u.y <= 0)
            return skeleton & domain


##################################################
##          BEZIER DISTANCE SOLVER              ##
##################################################

# Inspired by shadertoy.com/view/ltXSDB

# # Test if point p crosses line (a, b), returns sign of result
# def testCross(a, b, px, py):
#     return torch.sign((b.y-a.y) * (px-a.x) - (b.x-a.x) * (py-a.y))
#
#
# def mix(x, y, a):
#     return x*a+y*(1-a)
#
# # Determine which side we're on (using barycentric parameterization)
# def signBezier( A,  B,  C,  px, py):
#     a = C - A
#     b = B - A
#     cx, cy= px - A.x, py - A.y
#     bary = cx*b.y-b.x*cy , a.x*cy-cx*a.y
#     bary = [_ / (a.x*b.y-b.x*a.y) for _ in bary]
#     baryx, baryy = bary
#     d =  1.0 - baryx - baryy;
#     dx, dy = baryy * 0.5 + d, d
#     return mix(torch.sign(dx * dx - dy), mix(-1.0, 1.0,
#                         testCross(A, B, px, py) * testCross(B, C, px, py) > 0.0),
#                     ((d.x - d.y) > 0.0) * testCross(A, C, B.x, B.y))
#
#
# # Solve cubic equation for roots
# def solveCubic(a, b, c):
#     p = b - a*a / 3.0, p3 = p*p*p;
#     q = a * (2.0*a*a - 9.0*b) / 27.0 + c;
#     float d = q*q + 4.0*p3 / 27.0;
#     float offset = -a / 3.0;
#     if(d >= 0.0):
#         float z = sqrt(d);
#         vec2 x = (vec2(z, -z) - q) / 2.0;
#         vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
#         return vec3(offset + uv.x + uv.y);
#
#     float v = acos(-sqrt(-27.0 / p3) * q / 2.0) / 3.0;
#     float m = cos(v), n = sin(v)*1.732050808;
#     return vec3(m + m, -n - m, n - m) * sqrt(-p / 3.0) + offset;
#
#
# # Find the signed distance from a point to a bezier curve
# def sdBezier(A, B, C, px, py):
#     Bx = B.x if B.x * 2.0 - A.x - C.x != 0 else B.x + 1e-4
#     By = B.y if B.y * 2.0 - A.x - C.x != 0 else B.y + 1e-4
#     B = Point(Bx, By)
#
#     a = B - A
#     b = A - B * 2.0 + C
#     c = a * 2.0
#     dx, dy = A.x - px, A.y - py
#
#     # vec3 k = vec3(3.*dot(a,b), 2.*dot(a,a)+dot(d,b), dot(d,a)) / dot(b,b)
#     # vec3 t = clamp(solveCubic(k.x, k.y, k.z), 0.0, 1.0)
#     tx, ty = solveCubic(3*np.dot(a,b),
#                         2*np.dot(a,a) + dx*b.x+dy*b.y,
#                         (dx*a.x+dy*a.y) / np.dot(b,b))
#     tx, ty = [torch.clip(_, 0, 1) for _ in (tx, ty)]
#
#     pos = A + (c + b*tx)*tx
#     float dis = torch.sqrt(pos - p)
#     pos = A + (c + b*t.y)*t.y
#     dis = min(dis, length(pos - p))
#     pos = A + (c + b*t.z)*t.z
#     dis = min(dis, length(pos - p))
#     return dis * signBezier(A, B, C, p)
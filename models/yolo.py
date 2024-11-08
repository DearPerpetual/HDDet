import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  

from models.common import *
from models.experimental import *
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)
from utils.tal.anchor_generator import make_anchors, dist2bbox

try:
    import thop  
except ImportError:
    thop = None


class Detect(nn.Module):
    
    dynamic = False  
    export = False  
    shape = None
    anchors = torch.empty(0)  
    strides = torch.empty(0)  

    def __init__(self, nc=80, ch=(), inplace=True):  
        super().__init__()
        self.nc = nc  
        self.nl = len(ch)  
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  
        self.inplace = inplace  
        self.stride = torch.zeros(self.nl)  

        c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        
        m = self  # self.model[-1]  # Detect() module
        
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  


class DDetect(nn.Module):
   
    dynamic = False  
    export = False  
    shape = None
    anchors = torch.empty(0)  
    strides = torch.empty(0)  

    def __init__(self, nc=80, ch=(), inplace=True):  
        super().__init__()
        self.nc = nc  
        self.nl = len(ch)  
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  
        self.inplace = inplace  
        self.stride = torch.zeros(self.nl) 

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        
        m = self  # self.model[-1]  # Detect() module
        
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2) 


class DualDetect(nn.Module):
   
    dynamic = False  
    export = False  
    shape = None
    anchors = torch.empty(0)  
    strides = torch.empty(0)  

    def __init__(self, nc=80, ch=(), inplace=True):  
        super().__init__()
        self.nc = nc  
        self.nl = len(ch) // 2  
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  
        self.inplace = inplace  
        self.stride = torch.zeros(self.nl)  

        c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  
        c4, c5 = max((ch[self.nl] // 4, self.reg_max * 4, 16)), max((ch[self.nl], min((self.nc * 2, 128))))  
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 4 * self.reg_max, 1)) for x in ch[self.nl:])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  
        d1 = []
        d2 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
        if self.training:
            return [d1, d2]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2])

    def bias_init(self):
        
        m = self  # self.model[-1]  # Detect() module
        
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  
            a[-1].bias.data[:] = 1.0  
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  
            a[-1].bias.data[:] = 1.0  
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  


class DualDDetect(nn.Module):
    
    dynamic = False  
    export = False  
    shape = None
    anchors = torch.empty(0)  
    strides = torch.empty(0)  

    def __init__(self, nc=80, ch=(), inplace=True):  
        super().__init__()
        self.nc = nc  
        self.nl = len(ch) // 2  
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  
        self.inplace = inplace  
        self.stride = torch.zeros(self.nl)  

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  
        c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), max((ch[self.nl], min((self.nc * 2, 128))))  
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4), nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  
        d1 = []
        d2 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
        if self.training:
            return [d1, d2]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2])
  

    def bias_init(self):
        
        m = self  # self.model[-1]  
        
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  
            a[-1].bias.data[:] = 1.0  
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  
            a[-1].bias.data[:] = 1.0  
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  


class TripleDetect(nn.Module):
    
    dynamic = False  
    export = False  
    shape = None
    anchors = torch.empty(0)  
    strides = torch.empty(0)  

    def __init__(self, nc=80, ch=(), inplace=True):  
        super().__init__()
        self.nc = nc  
        self.nl = len(ch) // 3  
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  
        self.inplace = inplace  
        self.stride = torch.zeros(self.nl)  

        c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  
        c4, c5 = max((ch[self.nl] // 4, self.reg_max * 4, 16)), max((ch[self.nl], min((self.nc * 2, 128))))  
        c6, c7 = max((ch[self.nl * 2] // 4, self.reg_max * 4, 16)), max((ch[self.nl * 2], min((self.nc * 2, 128))))  
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 4 * self.reg_max, 1)) for x in ch[self.nl:self.nl*2])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:self.nl*2])
        self.cv6 = nn.ModuleList(
            nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3), nn.Conv2d(c6, 4 * self.reg_max, 1)) for x in ch[self.nl*2:self.nl*3])
        self.cv7 = nn.ModuleList(
            nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in ch[self.nl*2:self.nl*3])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)
        self.dfl3 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  
        d1 = []
        d2 = []
        d3 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
            d3.append(torch.cat((self.cv6[i](x[self.nl*2+i]), self.cv7[i](x[self.nl*2+i])), 1))
        if self.training:
            return [d1, d2, d3]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc), 1)
        dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1), torch.cat((dbox3, cls3.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2, d3])

    def bias_init(self):
        
        m = self  # self.model[-1]  # Detect() module
        
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  
            a[-1].bias.data[:] = 1.0  
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  
            a[-1].bias.data[:] = 1.0  
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  
        for a, b, s in zip(m.cv6, m.cv7, m.stride):  
            a[-1].bias.data[:] = 1.0  
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  


class TripleDDetect(nn.Module):
    
    dynamic = False  
    export = False  
    shape = None
    anchors = torch.empty(0)  
    strides = torch.empty(0)  

    def __init__(self, nc=80, ch=(), inplace=True):  
        super().__init__()
        self.nc = nc  
        self.nl = len(ch) // 3  
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  
        self.inplace = inplace  
        self.stride = torch.zeros(self.nl)  

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[0], min((self.nc * 2, 128))))  
        c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[self.nl], min((self.nc * 2, 128))))  
        c6, c7 = make_divisible(max((ch[self.nl * 2] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[self.nl * 2], min((self.nc * 2, 128))))  
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), 
                          nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4), 
                          nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:self.nl*2])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:self.nl*2])
        self.cv6 = nn.ModuleList(
            nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3, g=4), 
                          nn.Conv2d(c6, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*2:self.nl*3])
        self.cv7 = nn.ModuleList(
            nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in ch[self.nl*2:self.nl*3])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)
        self.dfl3 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  
        d1 = []
        d2 = []
        d3 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
            d3.append(torch.cat((self.cv6[i](x[self.nl*2+i]), self.cv7[i](x[self.nl*2+i])), 1))
        if self.training:
            return [d1, d2, d3]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc), 1)
        dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
       
        y = torch.cat((dbox3, cls3.sigmoid()), 1)
        return y if self.export else (y, d3)

    def bias_init(self):
        
        m = self  # self.model[-1]  # Detect() module
        
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  
            a[-1].bias.data[:] = 1.0  
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  
            a[-1].bias.data[:] = 1.0  
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  
        for a, b, s in zip(m.cv6, m.cv7, m.stride):  
            a[-1].bias.data[:] = 1.0  
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  


class Segment(Detect):
    
    def __init__(self, nc=80, nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, ch, inplace)
        self.nm = nm  
        self.npr = npr  
        self.proto = Proto(ch[0], self.npr, self.nm)  
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class DSegment(DDetect):
    
    def __init__(self, nc=80, nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, ch[:-1], inplace)
        self.nl = len(ch)-1
        self.nm = nm  
        self.npr = npr  
        self.proto = Conv(ch[-1], self.nm, 1)  
        self.detect = DDetect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch[:-1])

    def forward(self, x):
        p = self.proto(x[-1])
        bs = p.shape[0]

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  
        x = self.detect(self, x[:-1])
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class DualDSegment(DualDDetect):
    
    def __init__(self, nc=80, nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, ch[:-2], inplace)
        self.nl = (len(ch)-2) // 2
        self.nm = nm  
        self.npr = npr  
        self.proto = Conv(ch[-2], self.nm, 1)  
        self.proto2 = Conv(ch[-1], self.nm, 1)  
        self.detect = DualDDetect.forward

        c6 = max(ch[0] // 4, self.nm)
        c7 = max(ch[self.nl] // 4, self.nm)
        self.cv6 = nn.ModuleList(nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3), nn.Conv2d(c6, self.nm, 1)) for x in ch[:self.nl])
        self.cv7 = nn.ModuleList(nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nm, 1)) for x in ch[self.nl:self.nl*2])

    def forward(self, x):
        p = [self.proto(x[-2]), self.proto2(x[-1])]
        bs = p[0].shape[0]

        mc = [torch.cat([self.cv6[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2),
              torch.cat([self.cv7[i](x[self.nl+i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)]  
        d = self.detect(self, x[:-2])
        if self.training:
            return d, mc, p
        return (torch.cat([d[0][1], mc[1]], 1), (d[1][1], mc[1], p[1]))


class Panoptic(Detect):
    
    def __init__(self, nc=80, sem_nc=93, nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, ch, inplace)
        self.sem_nc = sem_nc
        self.nm = nm  
        self.npr = npr  
        self.proto = Proto(ch[0], self.npr, self.nm)  
        self.uconv = UConv(ch[0], ch[0]//4, self.sem_nc+self.nc)
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)


    def forward(self, x):
        p = self.proto(x[0])
        s = self.uconv(x[0])
        bs = p.shape[0]

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  
        x = self.detect(self, x)
        if self.training:
            return x, mc, p, s
        return (torch.cat([x, mc], 1), p, s) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p, s))
    

class BaseModel(nn.Module):
    
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  
        for m in self.model:
            if m.f != -1:  
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  
            y.append(x if m.i in self.save else None)  
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (RepConvN)) and hasattr(m, 'fuse_convs'):
                m.fuse_convs()
                m.forward = m.forward_fuse  
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  
                delattr(m, 'bn')  
                m.forward = m.forward_fuse  
        self.info()
        return self

    def info(self, verbose=False, img_size=640): 
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        
        self = super()._apply(fn)
        m = self.model[-1] 
        if isinstance(m, (Detect, DualDetect, TripleDetect, DDetect, DualDDetect, TripleDDetect, Segment, DSegment, DualDSegment, Panoptic)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
            
        return self


class DetectionModel(BaseModel):
   
    def __init__(self, cfg='yolo.yaml', ch=3, nc=None, anchors=None):  
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  
        else:  # is *.yaml
            import yaml  
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  

        
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  
        self.names = [str(i) for i in range(self.yaml['nc'])]  
        self.inplace = self.yaml.get('inplace', True)

        
        m = self.model[-1]  
        if isinstance(m, (Detect, DDetect, Segment, DSegment, Panoptic)):
            s = 256  
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, DSegment, Panoptic)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  
            
            self.stride = m.stride
            m.bias_init()  
        if isinstance(m, (DualDetect, TripleDetect, DualDDetect, TripleDDetect, DualDSegment)):
            s = 256  
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0][0] if isinstance(m, (DualDSegment)) else self.forward(x)[0]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  
           
            self.stride = m.stride
            m.bias_init()  

        
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  
        return self._forward_once(x, profile, visualize)  

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  
        s = [1, 0.83, 0.67]  
        f = [None, 3, None]  
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  
            
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  
        return torch.cat(y, 1), None  

    def _descale_pred(self, p, flips, scale, img_size):
        
        if self.inplace:
            p[..., :4] /= scale  
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  
            if flips == 2:
                y = img_size[0] - y  
            elif flips == 3:
                x = img_size[1] - x  
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        
        nl = self.model[-1].nl  
        g = sum(4 ** x for x in range(nl))  
        e = 1  
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  
        y[0] = y[0][:, :-i]  
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  
        y[-1] = y[-1][:, i:]  
        return y


Model = DetectionModel  


class SegmentationModel(DetectionModel):
    
    def __init__(self, cfg='yolo-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        
        if isinstance(model, DetectMultiBackend):
            model = model.model  
        model.model = model.model[:cutoff]  
        m = model.model[-1]  
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  
        c = Classify(ch, nc)  
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  
        model.model[-1] = c  
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        
        self.model = None


def parse_model(d, ch):  
    
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  
        RepConvN.default_act = eval(act)  
        LOGGER.info(f"{colorstr('activation:')} {act}")  
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  
    no = na * (nc + 5)  

    layers, save, c2 = [], [], ch[-1]  
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  
        m = eval(m) if isinstance(m, str) else m  
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  

        n = n_ = max(round(n * gd), 1) if n > 1 else n  
        if m in {
            Conv, AConv, ConvTranspose, 
            Bottleneck, SPP, SPPF, DWConv, BottleneckCSP, nn.ConvTranspose2d, DWConvTranspose2d, SPPCSPC, ADown,
            ELAN1, RepNCSPELAN4, SPPELAN, CoordAtt, GhostModule, GlobalContextBlock, EVCBlock, ShuffleAttention, ODConv2d, iRMB, AKConv}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, SPPCSPC}:
                args.insert(2, n)  
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        # TODO: channel, gw, gd
        elif m in {Detect, DualDetect, TripleDetect, DDetect, DualDDetect, TripleDDetect, Segment, DSegment, DualDSegment, Panoptic}:
            args.append([ch[x] for x in f])
            # if isinstance(args[1], int):  
            #     args[1] = [list(range(args[1] * 2))] * len(f)
            if m in {Segment, DSegment, DualDSegment, Panoptic}:
                args[2] = make_divisible(args[2] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  
        t = str(m)[8:-2].replace('__main__.', '')  
        np = sum(x.numel() for x in m_.parameters())  
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='model.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  
    print_args(vars(opt))
    device = select_device(opt.device)

    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)
    model.eval()

    if opt.line_profile:  
        model(im, profile=True)

    elif opt.profile:  
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  
        model.fuse()

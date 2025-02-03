from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise
# ===========================ECA模块添加实验=======================================
class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        m = x * y.expand_as(x)
        out = m + x
        return out


# ===========================ECA模块添加实验=======================================
# ===========================RCCA模块添加实验=======================================
def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 8
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()

        # [b, c', h, w]
        query = self.ConvQuery(x)
        # [b, w, c', h] -> [b*w, c', h] -> [b*w, h, c']
        query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        # [b, h, c', w] -> [b*h, c', w] -> [b*h, w, c']
        query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

        # [b, c', h, w]
        key = self.ConvKey(x)
        # [b, w, c', h] -> [b*w, c', h]
        key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        # [b, h, c', w] -> [b*h, c', w]
        key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # [b, c, h, w]
        value = self.ConvValue(x)
        # [b, w, c, h] -> [b*w, c, h]
        value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        # [b, h, c, w] -> [b*h, c, w]
        value_W = value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # [b*w, h, c']* [b*w, c', h] -> [b*w, h, h] -> [b, h, w, h]
        energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)
        # [b*h, w, c']*[b*h, c', w] -> [b*h, w, w] -> [b, h, w, w]
        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)
        # [b, h, w, h+w]  concate channels in axis=3

        concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))
        # [b, h, w, h] -> [b, w, h, h] -> [b*w, h, h]
        attention_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        attention_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w)

        # [b*w, h, c]*[b*w, h, h] -> [b, w, c, h]
        out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x

class RCCAModule(nn.Module):
    def __init__(self, recurrence=2, in_channels=1024):
        super(RCCAModule, self).__init__()
        self.recurrence = recurrence
        self.in_channels = in_channels
        self.inter_channels = in_channels // 4
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels)
        )
        self.CCA = CrissCrossAttention(self.inter_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.Conv2d(self.inter_channels, self.in_channels, 1,1)
        )

    def forward(self, x):
        output = self.conv_in(x)
        for i in range(self.recurrence):
            output = self.CCA(output)
        output = self.conv_out(output)
        output = output+x
        return output
# ===========================RCCA模块添加实验=======================================
# ===========================MultiScale模块添加实验=======================================
class MultiScale(nn.Module):
    def __init__(self,in_channels=1024,out_channels=1024):
        super(MultiScale, self).__init__()
        self.branch1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,dilation=1)
        self.branch2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=4, dilation=4)
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = torch.mean(x, dim=1, keepdim=True)
        out4 = out4.expand_as(out1)
        out = out1 + out2 + out3 +out4
        return out
# ===========================MultiScale模块添加实验=======================================
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)
        # build car head
        self.upsample = UpsampleBlock(256, 256)
        self.car_head = CARHead(cfg, 256)
        # build response map
        self.ECA = ECA(256)
        self.xcorr_depthwise = xcorr_depthwise
        self.RCCA = RCCAModule(2,768)
        self.MutiScale = MultiScale(256,256)
        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)
        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0], self.zf[0])
        features1 = self.upsample(features)
        features1 = 0.2 * features1 + 0.8 * xf[0]
        zf1 = self.ECA(self.zf[0])
        features1 = self.xcorr_depthwise(features1, zf1)
        features = features1 + features

        features_new1 = self.xcorr_depthwise(xf[1], self.zf[1])
        features_new11 = self.upsample(features_new1)
        features_new11 = 0.2 * features_new11 + 0.8 * xf[1]
        zf2 = self.ECA(self.zf[1])
        features_new11 = self.xcorr_depthwise(features_new11, zf2)
        features_new1 = features_new1 + features_new11

        features_new2 = self.xcorr_depthwise(xf[2], self.zf[2])
        features_new21 = self.upsample(features_new2)
        features_new21 = 0.2 * features_new21 + 0.8 * xf[2]
        zf3 = self.ECA(self.zf[2])
        features_new21 = self.xcorr_depthwise(features_new21, zf3)
        features_new2 = features_new2 + features_new21
        features = torch.cat([features, features_new1, features_new2], 1)
        features = self.RCCA(features)
        features = self.down(features)
        features = self.MutiScale(features)
        cls, loc, cen = self.car_head(features)
        return {
            'cls': cls,
            'loc': loc,
            'cen': cen
        }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0], zf[0])
        features_new1 = self.xcorr_depthwise(xf[1], zf[1])
        features_new2 = self.xcorr_depthwise(xf[2], zf[2])
        # features_new3 = self.xcorr_depthwise(xf[3], zf[3])
        features = torch.cat([features, features_new1, features_new2], 1)
        # [32, 256, 25, 25]
        features = self.RCCA(features)
        features = self.down(features)
        features = self.MutiScale(features)
        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs

import torch
import torch.nn as nn
import torch.nn.functional as F

def Norm2d(in_channels, **kwargs):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = nn.BatchNorm2d
    normalization_layer = layer(in_channels, **kwargs)
    return normalization_layer

def BNReLU(ch):
    return nn.Sequential(
           Norm2d(ch),
           nn.ReLU())

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.

        Output:
          The correlation of every class map with every feature map
          shape = [n, num_feats, num_classes, 1]


    """
    def __init__(self, cls_num=0, scale=1):
        super().__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, _, _ = probs.size(0), probs.size(1), probs.size(2), \
            probs.size(3)

        # each class image now a vector
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)

        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature
                            maps (save memory cost)
    Return:
        N X C X H X W
    '''
    def __init__(self, in_channels, key_channels, scale=1):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.in_channels),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear',
                                    align_corners=False)

        return context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation
    for each pixel.
    """
    def __init__(self, in_channels, key_channels, out_channels, scale=1,
                 dropout=0.1):
        super().__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels,
                                                         key_channels,
                                                         scale)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0,
                      bias=False),
            BNReLU(out_channels),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output

class OCRBlock(nn.Module):
    """
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """
    def __init__(self, in_channels,num_classes,key_channels=256,mid_channel=512):
        super().__init__()

        ocr_mid_channels = mid_channel
        ocr_key_channels = key_channels
        num_classes = num_classes

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(in_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BNReLU(ocr_mid_channels),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0,
            bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=1, stride=1, padding=0),
            BNReLU(in_channels),
            nn.Conv2d(in_channels, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats
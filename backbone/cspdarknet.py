import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        if act:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        return self.convs(x)


class ResidualBlock(nn.Module):
    """
    basic residual block for CSP-Darknet
    """
    def __init__(self, in_ch):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv(in_ch, in_ch, k=1)
        self.conv2 = Conv(in_ch, in_ch, k=3, p=1, act=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        h = self.conv2(self.conv1(x))
        out = self.act(x + h)

        return out


class CSPStage(nn.Module):
    def __init__(self, c1, n=1):
        super(CSPStage, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c1, c_, k=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(in_ch=c_) for _ in range(n)])
        self.cv3 = Conv(2 * c_, c1, k=1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.res_blocks(self.cv2(x))

        return self.cv3(torch.cat([y1, y2], dim=1))


# CSPDarknet
class CSPDarknet53(nn.Module):
    """
    CSPDarknet_53.
    """
    def __init__(self, num_classes=1000):
        super(CSPDarknet53, self).__init__()
            
        self.layer_1 = nn.Sequential(
            Conv(3, 32, k=3, p=1),      
            Conv(32, 64, k=3, p=1, s=2),
            CSPStage(c1=64, n=1)                       # p1/2
        )
        self.layer_2 = nn.Sequential(   
            Conv(64, 128, k=3, p=1, s=2),             
            CSPStage(c1=128, n=2)                      # P2/4
        )
        self.layer_3 = nn.Sequential(
            Conv(128, 256, k=3, p=1, s=2),             
            CSPStage(c1=256, n=8)                      # P3/8
        )
        self.layer_4 = nn.Sequential(
            Conv(256, 512, k=3, p=1, s=2),             
            CSPStage(c1=512, n=8)                      # P4/16
        )
        self.layer_5 = nn.Sequential(
            Conv(512, 1024, k=3, p=1, s=2),             
            CSPStage(c1=1024, n=4)                     # P5/32
        )


    def forward(self, x, targets=None):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def cspdarknet53(pretrained=False, hr=False, **kwargs):
    """Constructs a CSPDarknet_X model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CSPDarknet53()
    if pretrained:
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        if hr:
            print('Loading the cspdarknet53-448 ...')
            model.load_state_dict(torch.load(path_to_dir + '/weights/cspdarknet53/cspdarknet53_hr_76.9.pth', map_location='cuda'), strict=False)
        else:
            print('Loading the cspdarknet53 ...')
            model.load_state_dict(torch.load(path_to_dir + '/weights/cspdarknet53/cspdarknet53_75.7.pth', map_location='cuda'), strict=False)
    return model


if __name__ == '__main__':
    import time
    net = CSPDarknet53()
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    y = net(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
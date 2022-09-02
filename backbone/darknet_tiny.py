import torch
import torch.nn as nn


model_urls = {
    "darknet_tiny": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet_tiny.pth",
}


__all__ = ['darknet_tiny']


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet_Tiny(nn.Module):
    def __init__(self):
        
        super(DarkNet_Tiny, self).__init__()
        # backbone network : DarkNet_Tiny
        self.conv_1 = Conv_BN_LeakyReLU(3, 16, 3, 1)
        self.maxpool_1 = nn.MaxPool2d((2, 2), 2)              # stride = 2

        self.conv_2 = Conv_BN_LeakyReLU(16, 32, 3, 1)
        self.maxpool_2 = nn.MaxPool2d((2, 2), 2)              # stride = 4

        self.conv_3 = Conv_BN_LeakyReLU(32, 64, 3, 1)
        self.maxpool_3 = nn.MaxPool2d((2, 2), 2)              # stride = 8

        self.conv_4 = Conv_BN_LeakyReLU(64, 128, 3, 1)
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)              # stride = 16

        self.conv_5 = Conv_BN_LeakyReLU(128, 256, 3, 1)
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)              # stride = 32

        self.conv_6 = Conv_BN_LeakyReLU(256, 512, 3, 1)
        self.maxpool_6 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d((2, 2), 1)                           # stride = 32
        )

        self.conv_7 = Conv_BN_LeakyReLU(512, 1024, 3, 1)


    def forward(self, x):
        x = self.conv_1(x)
        c1 = self.maxpool_1(x)
        c1 = self.conv_2(c1)
        c2 = self.maxpool_2(c1)
        c2 = self.conv_3(c2)
        c3 = self.maxpool_3(c2)
        c3 = self.conv_4(c3)
        c4 = self.maxpool_4(c3)
        c4 = self.conv_5(c4)       # stride = 16
        c5 = self.maxpool_5(c4)  
        c5 = self.conv_6(c5)
        c5 = self.maxpool_6(c5)
        c5 = self.conv_7(c5)       # stride = 32

        output = {
            'layer1': c3,
            'layer2': c4,
            'layer3': c5
        }

        return output


def build_darknet_tiny(pretrained=False):
    # model
    model = DarkNet_Tiny()

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['darknet_tiny']
        # checkpoint state dict
        checkpoint_state_dict = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model


if __name__ == '__main__':
    import time
    net = build_darknet_tiny(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    output = net(x)
    t1 = time.time()
    print('Time: ', t1 - t0)

    for k in output.keys():
        print('{} : {}'.format(k, output[k].shape))

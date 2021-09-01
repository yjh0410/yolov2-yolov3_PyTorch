import torch
from thop import profile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def FLOPs_and_Params(model, size):
    x = torch.randn(1, 3, size, size).to(device)
    model.trainable = False
    model.eval()

    flops, params = profile(model, inputs=(x, ))
    print('FLOPs : ', flops / 1e9, ' B')
    print('Params : ', params / 1e6, ' M')

    model.trainable = True
    model.train()


if __name__ == "__main__":
    pass

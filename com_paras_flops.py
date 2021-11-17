import torch
from thop import profile
from models.yolov3 import YOLOv3 as net_1



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    input_image = torch.randn(1, 3, 608, 608).to(device)
    input_size = 608
    num_classes = 80
    anchor_size =  [[32.64, 47.68], [50.24, 108.16], [126.72, 96.32],     
                        [78.4, 201.92], [178.24, 178.56], [129.6, 294.72],
                        [78.4, 201.92], [178.24, 178.56], [129.6, 294.72]]
    model = net_1(device, input_size=input_size, num_classes=num_classes, trainable=False, anchor_size=anchor_size).to(device)
    flops, params = profile(model, inputs=(input_image, ))
    print('FLOPs : ', flops / 1e6, ' M')
    print('Params : ', params / 1e6, ' M')


if __name__ == "__main__":
    main()

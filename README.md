# pytorch-yolo-v2
I really enjoy yolo. It is so amazing! So I try to reproduce it. And I almost got it!

Give me some time, and I will write a complete README to teach everyone how to use this project.
Of course, if your coding is excellent, you can easily know how to use this project.

Otherwise, my yolo-v2 got 72.2 mAP on VOC2007 test with input resolution 416 image size, lower than origin yolo-v2 that got 76.8% mAP with the same image size. This is maybe because that there are two tricks I didn't use:

1. hi-res classifier

The darknet19-448 is being trained on ImageNet, so just hold on.

2. multi-scale train

My dataloader isn't suitable for this trick. But I'm trying to change it. After that, I will try this trick.

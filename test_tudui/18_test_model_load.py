import torchvision
import torch

# # 加载方式1 保存网络模型结构和参数
# model = torch.load("vgg16_mothod1.pth")
# print(model)

# 加载方式2 
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_mothod2.pth"))
# model = torch.load("vgg16_mothod2.pth")
print(vgg16)
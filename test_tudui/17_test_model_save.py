import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1 保存网络模型结构和参数
torch.save(vgg16, "vgg16_mothod1.pth")

# 保存方式2(官方推荐) 只保存参数，保存成字典的格式
torch.save(vgg16.state_dict(), "vgg16_mothod2.pth")

# 陷阱：自定义模型用方式1时，加载时，需要把模型定义写到load的文件里
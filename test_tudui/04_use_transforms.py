from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2

writer = SummaryWriter("logs") 

img_path = "../hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img)

# ToTensor
trans_tesnor = transforms.ToTensor() 
img_tensor = trans_tesnor(img)  
writer.add_image("ToTensor",img_tensor) 

# Normalize
print(img_tensor[0][0][0])
# Normalize(mean, std)
tensor_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) #input[channel]=(input[chnnel]-mean[channel])/std[channel]            
img_norm = tensor_norm(img_tensor)  
print(img_norm[0][0][0])

# Resize-1
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_tesnor(img_resize)
writer.add_image("Resize",img_tensor, 0) 
print(img_resize)

# Resize-2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_tesnor]) # Compose函数中后面一个参数的输入为前面一个参数的输出  
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2, 1) 
print(img_resize_2)

"""
compose相当于把两个函数合并为一个函数

img_resize_2 = trans_compose(img) 
和
img_resize_2 = trans_resize_2(img)
img_resize_2 = trans_tesnor(img_resize_2)
等价
"""
writer.close()
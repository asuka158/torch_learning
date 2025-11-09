from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2

img_path = "../hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path) # PIL读取的是RGB模式

writer = SummaryWriter("logs") 

tensor_trans = transforms.ToTensor() 
tensor_img = tensor_trans(img)  # 转为tensor后，通道顺序不变

# opencv读取，类型是numpy.ndarray
cv_img = cv2.imread(img_path) # opencv读取的是，BGR模式
# print(type(cv_img))
# print(cv_img.shape)
cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) # 转为RGB模式
writer.add_image("cv_img_corrected", cv_img_rgb, dataformats="HWC")

writer.add_image("Temsor_img",tensor_img) 
writer.add_image("cv_img",cv_img, dataformats="HWC") 
writer.close()
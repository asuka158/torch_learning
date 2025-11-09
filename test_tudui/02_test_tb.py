from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

# for i in range(100):
#     writer.add_scalar("y = 2x", 2 * i, i)

img_path1 = "../hymenoptera_data/train/ants/0013035.jpg" 
img_PIL1 = Image.open(img_path1)
img_array1 = np.array(img_PIL1)

img_path2 = "../hymenoptera_data/train/bees/17209602_fe5a5a746f.jpg" 
img_PIL2 = Image.open(img_path2)
img_array2 = np.array(img_PIL2)

# 查看数据类型，查看shape，dataformats默认是CHW，如果不是的话需要额外指定
# print(type(img_array1))
# print(img_array1.shape)

writer.add_image("test",img_array1,1,dataformats="HWC") # 1 表示该图片在第1步
writer.add_image("test",img_array2,2,dataformats="HWC") # 2 表示该图片在第2步                   

writer.close()


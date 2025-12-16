import torchvision
from PIL import Image
from model import *

image_path = "./airplane.png"
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])

image = transform(image)
print(image)
print(image.shape)

model = torch.load("tudui_29.pth", map_location=torch.device('cpu'), weights_only=False) # python 20_test_train_gpu.py 获取 
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
print(image.shape)
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))
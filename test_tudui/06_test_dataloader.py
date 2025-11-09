import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data # 每个data都是由4张图片组成，imgs.size 为 [4,3,32,32]，四张32×32图片三通道，targets由四个标签组成             
        writer.add_images("Epoch：{}".format(epoch),imgs,step)
        step = step + 1
    
writer.close()
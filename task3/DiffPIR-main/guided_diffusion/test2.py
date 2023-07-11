import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# 数据集
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

# 定义数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=1)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=1)

# 数据类
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


# 函数用于展示图片
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(double_conv2d_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.layer1_conv = double_conv2d_bn(1, 8)
        self.layer2_conv = double_conv2d_bn(8, 16)
        self.layer3_conv = double_conv2d_bn(16, 32)
        self.layer4_conv = double_conv2d_bn(32, 64)
        self.layer5_conv = double_conv2d_bn(64, 128)
        self.layer6_conv = double_conv2d_bn(128, 64)
        self.layer7_conv = double_conv2d_bn(64, 32)
        self.layer8_conv = double_conv2d_bn(32, 16)
        self.layer9_conv = double_conv2d_bn(16, 8)
        self.layer10_conv = nn.Conv2d(8, 1, kernel_size=3,
                                      stride=1, padding=1, bias=True)

        self.deconv1 = deconv2d_bn(128, 64)
        self.deconv2 = deconv2d_bn(64, 32)
        self.deconv3 = deconv2d_bn(32, 16)
        self.deconv4 = deconv2d_bn(16, 8)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1, 2)

        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)

        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        outp = self.sigmoid(outp)
        return outp





if __name__ == "__main__":
 net = Net()
 criterion = nn.CrossEntropyLoss()
 optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
 from torch.utils.tensorboard import SummaryWriter

# 在没有指定要写入文件默认写入到 `log_dir` 是 "runs"
 writer = SummaryWriter('runs/fashion_mnist_experiment_1')
 '''
 print(net)
 y=torch.rand(1,1,256)
 print((trainloader))
 for i,data in enumerate(trainloader):
     if i==1:
        break
     else:
        imgs,label=data
        print(data)
        print(imgs)
        print(imgs.size())'''
 model = Unet()
 inp = torch.rand(10, 1, 224, 224)
 outp = model(inp)

 writer.add_graph(model, inp)
 writer.close()

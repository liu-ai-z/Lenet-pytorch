import torch.nn as nn
import torch.nn.functional as F
#tensor参数[batch，channel，height，width]
#batch图片输入个数
#channel 通道数
#height图片高度
#width图片宽度

#N = ( W - F+2P)/s +1
# 输入图片大小W*W
# Filter大小F*F
# 步长s
# padding的像素P

class LeNet (nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()#super为了调用父类方法
        self.conv1 = nn.Conv2d(3,16,5) #16*220*220
        self.pool1 = nn.MaxPool2d(2,2) #16*110*110
        self.conv2 = nn.Conv2d(16,32,5)#32*106*106
        self.pool2 = nn.MaxPool2d(2,2)#32*53*53
        self.fc1 = nn.Linear(32 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward (self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1,32 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


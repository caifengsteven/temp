import torch
import torch.nn as nn
import torch.nn.functional as F


class OrderbookCNN(nn.Module):
    def __init__(self, input_size):
        super(OrderbookCNN, self).__init__()
        layers_params = [
            [1, 256, (6, 6), (0, 5), (1, 2)],  # conv1
            [256, 512, (2, 2), (1, 1), (1, 1)],  # conv2
            [512, 1024, (2, 2), (1, 1), (1, 1)]  # conv3
        ]
        self.layers = nn.ModuleList()
        self.poolkernel = [(2,2),(2,1),(2,1)]  # 提取padding信息

        # 根据输入尺寸和第一层参数计算输出尺寸
        input_h, input_w = input_size
        outh, outw = input_size
        # 构建卷积层
        for i,(in_channels, out_channels, kernel_size, padding, stride) in enumerate(layers_params):
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.LeakyReLU(negative_slope=0.01))
            outh, outw = self.calculate_output_size(outh, outw, (in_channels, out_channels, kernel_size, padding, stride))
            k = self.poolkernel[i]
            self.layers.append(nn.MaxPool2d(kernel_size=k))
            outh = outh//k[0]
            outw = outw//k[1]
        # 添加Dropout和全连接层
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1024 * outh * outw, 3)  # 假设最后的卷积层输出通道数为1024

    def calculate_output_size(self, input_h, input_w, layer_param):
        in_channels, out_channels, kernel_size, padding, stride = layer_param
        outh = ((input_h + 2 * padding[0] - kernel_size[0]) // stride[0]) + 1
        outw = ((input_w + 2 * padding[1] - kernel_size[1]) // stride[1]) + 1
        return outh, outw

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # 展平除了批次维度的所有维度
        x = x.view(x.size(0), -1)
        # 应用Dropout
        x = self.dropout(x)
        # 全连接层
        x = self.fc(x)
        return x

img_size = (41,30)
x = torch.rand(img_size).reshape(1,1,img_size[0],img_size[1])
# Create the model
model = OrderbookCNN(img_size)

y = model(x)
print(y.shape)
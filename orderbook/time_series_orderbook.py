import torch
import numpy as np
import matplotlib.pyplot as plt
from model import OrderbookCNN
# 假设当前的5档买单和5档卖单数据

orderbooks = [ [
    (25.20, 100), (25.21, 150), (25.22, 200), (25.23, 250), (25.24, 300),  # Bids
    (25.25, 350), (25.26, 300), (25.27, 250), (25.28, 200), (25.29, 150)   # Asks
],
[
    (25.20, 100), (25.21, 150), (25.22, 200), (25.23, 250), (25.26, 300),  # Bids
    (25.27, 350), (25.28, 300), (25.29, 250), (25.30, 200), (25.31, 400)   # Asks
],
[
    (25.16, 100), (25.17, 150), (25.18, 200), (25.19, 250), (25.20, 300),  # Bids
    (25.21, 350), (25.22, 300), (25.24, 250), (25.25, 300), (25.26, 450)   # Asks
],
[
    (25.12, 500), (25.14, 550), (25.15, 300), (25.16, 250), (25.18, 300),  # Bids
    (25.19, 350), (25.20, 300), (25.21, 250), (25.22, 200), (25.23, 550)   # Asks
],

]

# 历史前n个orderbook的最大挂单量
maxvolume = 500

# 参数设置
M = 10  # 价格水平的像素高度
I = 1
delta = 0.01 #价差颗粒度
# 计算图像的高度，每个价格水平有M个像素点
image_height = 2*I*M + 1  # 21
image_width = 3       # 3列，中间为空白


bg = np.full(((image_height-1)*2+1, image_width*10), 0)

start = (bg.shape[0]-1)//2 - (image_height-1)//2

start_y = start



last_mid = None

for j in range(len(orderbooks)):
    od  = orderbooks[j]
    mid = len(od) // 2
    mid_price = od[mid - 1][0]  # +a[mid][0]

    # 初始化图像数组，中间列初始化为255（白色），两侧可以设为0（黑色）
    image = np.full((image_height, image_width), 0)
    image[:, 1] = 255

    # 将LOB数据转换为图像
    # 计算每个价格水平在图像中的位置
    bid_start = (image_height - 1) // 2 + 1
    ask_start = (image_height - 1) // 2 - 1

    for i, (price, volume) in enumerate(od):
        # 计算归一化后的价格索引高度
        # 买单从图像底部开始，卖单从顶部开始
        K = int((price - mid_price) / delta)
        if i < len(od) // 2:
            # 买单
            pixel_index = bid_start - K * I
            image[pixel_index, 0] = image[pixel_index, 0] + volume
        else:
            # 卖单
            pixel_index = ask_start - K * I
            image[pixel_index, 2] = image[pixel_index, 2] + volume  # 映射到右侧

    image[:, 0] = image[:, 0] / max(maxvolume, image[:, 0].max()) * 255
    image[:, 2] = image[:, 2] / max(maxvolume, image[:, 2].max()) * 255

    #赋值到bg
    start_y = start if j==0 else start_y - int((mid_price-last_mid)/delta)
    start_x = j*3
    last_mid = mid_price
    bg[start_y:start_y+image_height,start_x:start_x+3] = image

plt.imshow(bg, cmap='gray')
plt.colorbar()
plt.show()

bg = bg/255.
x = torch.from_numpy(bg).reshape(1,1,bg.shape[0],bg.shape[1]).float()
# Create the model
model = OrderbookCNN(bg.shape)

y = model(x)
print(y) # 1*3 up neutral down y预测未来的涨跌

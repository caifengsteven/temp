import numpy as np
import matplotlib.pyplot as plt

# 假设当前的5档买单和5档卖单数据
a = [
    (25.20, 100), (25.21, 150), (25.22, 200), (25.23, 250), (25.24, 300),  # Bids
    (25.25, 350), (25.26, 300), (25.27, 250), (25.28, 200), (25.29, 150)   # Asks
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

mid = len(a) // 2
mid_price = a[mid-1][0] #+a[mid][0]

# 初始化图像数组，中间列初始化为255（白色），两侧可以设为0（黑色）
image = np.full((image_height, image_width), 0)
image[:,1] = 255

# 将LOB数据转换为图像
# 计算每个价格水平在图像中的位置
bid_start =  (image_height-1)//2+1
ask_start = (image_height-1)//2-1



for i, (price, volume) in enumerate(a):
    # 计算归一化后的价格索引高度
    # 买单从图像底部开始，卖单从顶部开始
    if i < len(a) // 2:
        # 买单
        K = int((price-mid_price)/delta)
        pixel_index = bid_start-K*I
        image[pixel_index, 0] =  image[pixel_index, 0]+volume
    else:
        # 卖单
        K = int((price-mid_price)/delta)
        pixel_index = ask_start-K*I
        image[pixel_index, 2] = image[pixel_index, 0]+volume  # 映射到右侧

image[:, 0] = image[:, 0]/max(maxvolume,image[:, 0].max())*255
image[:, 2] = image[:, 2]/max(maxvolume,image[:, 2].max())*255
print(image)
# 显示图像
plt.imshow(image, cmap='gray')
plt.colorbar()
plt.show()


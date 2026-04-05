### 论文简述

根据Elliott waves理论，股票的K线波动有12种pattern

![img](file:///C:/Users/Aaron/AppData/Local/Temp/msohtmlclip1/01/clip_image001.png)

论文的思路是这样的，首先使用一个20-16-12的多分类神经网络对12种pattern进行识别，然后使用一个12-6-2的神经网络对价格的趋势进行预测（涨或者跌），预测的依据是一个经验模型，即k线呈现出某种pattern的Eillot wave时，其趋势是涨或跌。

分类网络的训练数据就是12种pattern；

预测网络的训练数据是下面这张表；

![计算机生成了可选文字: trainingsetforthesecondneuralnetwork. Inputneurons 私粽糅 Outputneurons](file:///C:/Users/Aaron/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)

其他网络训练的超参数文中都已经给出。

 

### Roadmap 

实现方式为python + tensorflow

1 分类网络和预测网络的代码实现 （4h）

2 自定义pattern生成工具 （3h）

2 训练数据准备，包含分类网络的训练数据和预测网络的数据（2h）

3 两个网络的训练（8h）

4 使用客户提供的数据进行测试，给出测试结果，参考论文里面的图，画一些类似的图（8h）
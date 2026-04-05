# 1 目录结构说明
├─cfg 配置文件目录，该目录保存了分类网络和预测网络  
├─exp_result 实验结果输出的目录  
├─log  
│  ├─classify_net 分类网络训练记录和模型存储目录  
│  └─predict_net 预测网络训练记录和模型存储目录  
├─Stk_Day_QFQ 实验数据目录  
├─train_data  训练数据目录  
│  └─Pattern 自定义Pattern保存目录  

# 运行环境配置说明
_这里的配置方法只针对windows，mac和Linux下的配置方法大同小异，由于我这边没有相应的环境，暂不演示。_

## 1.1 Python 3.6+ 

[下载链接](https://www.python.org/getit/)

安装的时候请勾选 Add Python to PATH

![1531637159634](C:\Users\Aaron\AppData\Local\Temp\1531637159634.png)

## 1.2  安装所需的第三方包

项目目录下面，打开console，输入

```python
pip install -r requirements.txt
```

这一步依网络状况不同需要等待的时间也不同，请耐心等待。

![1531642002755](C:\Users\Aaron\AppData\Local\Temp\1531642002755.png)

# 2 自定义Pattern
运行Pattern生成器

```
python PatternGenerator.py
```

Pattern生成器支持10个数据点，每个数据点可以**鼠标左键点选拖动调整点位的值**，调整完成后点击Save Pattern即可保存。

![1531637433105](C:\Users\Aaron\AppData\Local\Temp\1531637433105.png)

![1531637533297](C:\Users\Aaron\AppData\Local\Temp\1531637533297.png)

自定义的Pattern保存在Pattern目录下，编号依次增长，例如上图中，现在已有12个Pattern，新增一个Pattern可以命名为13



# 3 训练

## 3.1 网络结构调整 

分类网络和预测网络的配置分别在cfg/classify_net.cfg, cfg/predict_net.cfg中，两个文件可以用任意文本编辑器打开，其中定义了输入层、隐藏层和输出层的神经元个数，以classify_net.cfg为例：

```
input 20
layer1 16
output 12
```

其输入层为20，隐藏层为16，输出层为12，**输出层的12神经元对应12个Pattern，因此如果有新增的自定义Pattern，例如现在有13个Pattern了，拿输出层的个数也要改为13，如果自定义Pattern数量较大，输出层神经元数量比较大，可以考虑适当增加隐藏层神经元的个数**

## 3.2 训练数据准备

训练数据都在Pattern这个文件夹中，Pattern文件夹中包含了以数字编号命名的Pattern数据和趋势数据。

* Pattern数据由Pattern生成器生成，准备数据时不用动；
* 趋势数据的文件名是trend.txt，其内容如下。其中0表示Up，1表示Down，每一行对应一个Pattern的后续趋势，这一部分与论文原文的Table 2。**需要注意的是，这里12行数据，严格对应Pattern1到Pattern12，Pattern文件保存的时候需要严格 [编号].txt的格式命名，否则会导致Pattern数据和其趋势定义不匹配的问题，导致后面的训练没有意义。*如果有新增的自定义Pattern，trend.txt里也要新增一行***

```
0
1
0
0
1
1
0
1
0
1
0
0
```

## 3.3 训练

这里使用Python的notebook来一边训练一边展示训练的结果，这样比较方便查看；

首先在项目目录输入

```
jupyter-notebook
```

输入之后会自动打开浏览器加载一个网页如下，打开train.ipynb

![1531639066610](C:\Users\Aaron\AppData\Local\Temp\1531639066610.png)

选择kernal/Restart & Run All 即可开始一次训练

![1531639127890](C:\Users\Aaron\AppData\Local\Temp\1531639127890.png)

运行完成后网页里会同时显示训练的结果

![1531639305183](C:\Users\Aaron\AppData\Local\Temp\1531639305183.png)

**训练参数调整**

训练参数主要包括max_epoch和end_loss，即最大训练步数和训练终止条件

默认值定义在model.py文件的73行

![1531639428125](C:\Users\Aaron\AppData\Local\Temp\1531639428125.png)

如需更改，可以在notebook中添加代码指定，如

![1531639556661](C:\Users\Aaron\AppData\Local\Temp\1531639556661.png)

其他详细信息可以参考model.py文件中的代码注释

**训练好的模型分别保存在log/classify_net和log/predict_net中，每次训练都会覆盖这两个文件夹，如需长时间保存某次训练的结果，可以复制这两个文件夹**

## 3.4 使用已训练好的模型对数据进行实验

**预先将Stk_Day_QFQ 文件夹拷到项目目录下**

![1531639984796](C:\Users\Aaron\AppData\Local\Temp\1531639984796.png)

同样在notebook中，点击打开exp.ipynb

![1531639824583](C:\Users\Aaron\AppData\Local\Temp\1531639824583.png)

选择kernal/Restart & Run All 即可用训练好的模型跑一遍所有的数据，

![1531639851681](C:\Users\Aaron\AppData\Local\Temp\1531639851681.png)

这个notebook执行到最后一步的时候，会绘制所有的相似度大于给定阈值的样本数据，时间会很长，如果不需要的话可以手动点击下图中的停止按钮停止，也可以选择注释掉代码。

![1531640114952](C:\Users\Aaron\AppData\Local\Temp\1531640114952.png)

**加载特定的训练结果**

![1531640196787](C:\Users\Aaron\AppData\Local\Temp\1531640196787.png)

代码中定义神经网络变量的时候，设load_checkpoint=True，程序会自动从默认路径（log/classify_net和log/predict_net）加载已训练好的网络，如果要加载某个特定的训练结果，只需设值checkpoint_path的路径就行。

![1531640389948](C:\Users\Aaron\AppData\Local\Temp\1531640389948.png)



# 写在后面

说明一下，代码里是base 0的，即从0开始数，因此1号pattern在代码里会是0号，客户在阅读代码的时候请注意。另外，所有的图表输出都是有按base1输出的。
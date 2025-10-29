# Quick Start Guide - Curved Radius Supertrend

## 快速开始指南

### 1. 安装依赖 (Install Dependencies)

```bash
pip install -r requirements.txt
```

需要的包：
- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0

### 2. 运行测试 (Run Tests)

验证安装是否成功：

```bash
python test_indicator.py
```

应该看到：
```
==================================================
ALL TESTS PASSED ✓
==================================================
```

### 3. 运行简单示例 (Run Simple Example)

```bash
python example_usage.py
```

这将：
- 生成示例价格数据
- 计算指标
- 显示交易信号
- 创建可视化图表（保存为 `example_output.png`）

### 4. 查看参数对比 (View Parameter Comparison)

```bash
python visualize_indicator.py
```

这将创建一个包含6种不同参数配置的对比图（保存为 `curved_supertrend_comparison.png`）

### 5. 在您的代码中使用 (Use in Your Code)

```python
from curved_radius_supertrend import CurvedRadiusSupertrend
import numpy as np

# 准备您的数据
high = np.array([...])   # 最高价
low = np.array([...])    # 最低价
close = np.array([...])  # 收盘价

# 创建指标
indicator = CurvedRadiusSupertrend(
    atr_period=10,
    atr_multiplier=3.0,
    radius_strength=0.5,
    smoothness=3
)

# 计算
result = indicator.calculate(high, low, close)

# 获取当前趋势
current_trend = result['direction'].iloc[-1]
if current_trend == 1:
    print("上升趋势 (Uptrend)")
else:
    print("下降趋势 (Downtrend)")

# 获取趋势线值
trend_line = result['trend_line'].iloc[-1]
print(f"趋势线: {trend_line:.2f}")
```

### 6. 生成交易信号 (Generate Trading Signals)

```python
# 检测趋势变化
for i in range(1, len(result)):
    if result['direction'].iloc[i] != result['direction'].iloc[i-1]:
        if result['direction'].iloc[i] == 1:
            print(f"买入信号 (BUY) at bar {i}")
        else:
            print(f"卖出信号 (SELL) at bar {i}")
```

### 7. 参数调整建议 (Parameter Tuning)

根据您的交易风格选择参数：

**剥头皮 (Scalping) - 1-5分钟:**
```python
CurvedRadiusSupertrend(
    atr_period=7,
    atr_multiplier=2.0,
    radius_strength=0.2,  # 低曲率 = 更紧密的曲线
    smoothness=2
)
```

**日内交易 (Day Trading) - 15-60分钟:**
```python
CurvedRadiusSupertrend(
    atr_period=10,
    atr_multiplier=3.0,
    radius_strength=0.5,  # 中等曲率
    smoothness=3
)
```

**波段交易 (Swing Trading) - 4小时-日线:**
```python
CurvedRadiusSupertrend(
    atr_period=14,
    atr_multiplier=3.5,
    radius_strength=1.0,  # 高曲率 = 更宽的弧线
    smoothness=5
)
```

**趋势交易 (Position Trading) - 日线-周线:**
```python
CurvedRadiusSupertrend(
    atr_period=20,
    atr_multiplier=4.0,
    radius_strength=2.0,  # 很高的曲率
    smoothness=7
)
```

### 8. 理解输出 (Understanding Output)

指标返回的DataFrame包含：

| 列名 | 说明 | 值 |
|------|------|-----|
| `curved_upper` | 上轨曲线 | 浮点数 |
| `curved_lower` | 下轨曲线 | 浮点数 |
| `direction` | 趋势方向 | 1 (上升) 或 -1 (下降) |
| `trend_line` | 活动趋势线 | 浮点数 |

### 9. 可视化您的数据 (Visualize Your Data)

```python
import matplotlib.pyplot as plt

# 绘制价格和指标
plt.figure(figsize=(14, 7))
plt.plot(close, label='价格', color='black')
plt.plot(result['curved_upper'], label='上轨', color='red', linestyle='--')
plt.plot(result['curved_lower'], label='下轨', color='green', linestyle='--')
plt.plot(result['trend_line'], label='趋势线', linewidth=2)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 10. 常见问题 (FAQ)

**Q: 如何调整指标的灵敏度？**  
A: 降低 `radius_strength` 使其更灵敏，提高使其更平滑。

**Q: 如何减少假信号？**  
A: 增加 `smoothness` 参数或提高 `atr_multiplier`。

**Q: 指标适合什么市场？**  
A: 适合趋势市场。在横盘市场中可能产生较多假信号。

**Q: 可以用于加密货币吗？**  
A: 可以！适用于任何有OHLC数据的市场。

**Q: 如何与其他指标结合？**  
A: 可以与RSI、MACD等指标结合使用，用于确认信号。

### 11. 文件说明 (File Description)

- `curved_radius_supertrend.py` - 核心指标实现
- `visualize_indicator.py` - 可视化工具
- `test_indicator.py` - 测试套件
- `example_usage.py` - 简单示例
- `README.md` - 详细英文文档
- `实现说明.md` - 详细中文说明
- `QUICKSTART.md` - 本快速开始指南

### 12. 获取帮助 (Get Help)

查看详细文档：
- 英文：`README.md`
- 中文：`实现说明.md`

运行示例代码了解更多用法。

---

## 开始交易！ (Start Trading!)

现在您已经准备好使用曲率半径超级趋势指标了！

记住：
- ⚠️ 始终进行回测
- ⚠️ 使用适当的风险管理
- ⚠️ 不要仅依赖单一指标
- ⚠️ 本指标仅用于教育目的

祝交易顺利！ Good luck with your trading!


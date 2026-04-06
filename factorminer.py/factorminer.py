# -*- coding: utf-8 -*-

import os
import time
import re
import numpy as np
import pandas as pd
import yfinance as yf
import google.generativeai as genai
from scipy.stats import spearmanr
from datetime import datetime, timedelta

# ==========================================
# 1. 配置区域 (Configuration)
# ==========================================

# 请在此处替换为你的 Google Gemini API Key
# 如果是在 Colab 或支持的环境中，可以直接读取 os.environ
API_KEY = os.getenv("GOOGLE_API_KEY", "")

# 股票池 (使用美股科技巨头作为示例)
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'NVDA', 'AMZN', 'META', 'TSLA', 'AMD']
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"

# 挖掘参数
ITERATIONS = 3          # 挖掘轮数 (Ralph Loop 循环次数)
FACTORS_PER_BATCH = 5   # 每轮让 LLM 生成的因子数量
IC_THRESHOLD = 0.03     # IC 阈值 (论文中通常较小，如 0.02-0.05)
CORR_THRESHOLD = 0.7    # 相关性阈值 (超过此值则视为冗余)

# ==========================================
# 2. 数据获取与预处理 (Data Layer)
# ==========================================

def get_real_data(tickers, start, end):
    """
    获取真实市场数据 (yfinance)
    """
    print(f"正在下载数据: {tickers} ...")
    data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)

    # 重构数据格式: MultiIndex -> Dict of DataFrames (Open, High, Low, Close, Volume)
    # 我们需要的数据形状是: Index=Time, Columns=Stocks
    market_data = {}
    features = ['Open', 'High', 'Low', 'Close', 'Volume']

    # yfinance 的数据结构在不同版本可能不同，这里做适配
    # 假设结构是 (Date, (Ticker, Feature)) 或 (Date, (Feature, Ticker))

    df_close = pd.DataFrame()
    df_open = pd.DataFrame()
    df_high = pd.DataFrame()
    df_low = pd.DataFrame()
    df_volume = pd.DataFrame()

    for t in tickers:
        try:
            # 尝试获取每个 ticker 的数据
            stock_data = data[t]
            df_close[t] = stock_data['Close']
            df_open[t] = stock_data['Open']
            df_high[t] = stock_data['High']
            df_low[t] = stock_data['Low']
            df_volume[t] = stock_data['Volume']
        except KeyError:
            print(f"Warning: Could not extract data for {t}")
            continue

    # 填充缺失值
    market_data['close'] = df_close.ffill().bfill()
    market_data['open'] = df_open.ffill().bfill()
    market_data['high'] = df_high.ffill().bfill()
    market_data['low'] = df_low.ffill().bfill()
    market_data['volume'] = df_volume.ffill().bfill()

    # 计算 VWAP (近似计算: (High+Low+Close)/3 * Volume 的累加 / Volume 累加)
    # 简单起见，这里用 Typical Price 代替 VWAP 的瞬时值
    market_data['vwap'] = (market_data['high'] + market_data['low'] + market_data['close']) / 3.0

    # 计算收益率 (Target) - 下一期收益率
    market_data['returns'] = market_data['close'].pct_change()
    # Target 是 T+1 的收益率
    market_data['target'] = market_data['returns'].shift(-1)

    print("数据下载与预处理完成。")
    return market_data

# ==========================================
# 3. 算子库 (Skill Layer - Operator Library)
# ==========================================

class Operators:
    """
    对应论文中的 Operator Library。
    所有函数输入输出均为 DataFrame (Index=Time, Columns=Assets)
    """
    @staticmethod
    def Add(x, y): return x + y
    @staticmethod
    def Sub(x, y): return x - y
    @staticmethod
    def Mul(x, y): return x * y
    @staticmethod
    def Div(x, y): return x / (y + 1e-9) # 防止除零
    @staticmethod
    def Neg(x): return -x
    @staticmethod
    def Abs(x): return x.abs()
    @staticmethod
    def Log(x): return np.log(x.abs() + 1e-9)
    @staticmethod
    def Sign(x): return np.sign(x)

    @staticmethod
    def TsRank(x, window):
        """时间序列排序: 过去 window 天中当前值的排名"""
        return x.rolling(window).rank()

    @staticmethod
    def CsRank(x):
        """截面排序: 当前时间点各股票的排名 (0~1)"""
        return x.rank(axis=1, pct=True)

    @staticmethod
    def Delay(x, window):
        return x.shift(window)

    @staticmethod
    def Delta(x, window):
        return x - x.shift(window)

    @staticmethod
    def Std(x, window):
        return x.rolling(window).std()

    @staticmethod
    def Mean(x, window):
        return x.rolling(window).mean()

    @staticmethod
    def Max(x, window):
        return x.rolling(window).max()

    @staticmethod
    def Min(x, window):
        return x.rolling(window).min()

    @staticmethod
    def IfElse(condition, true_val, false_val):
        """
        如果 condition > 0 则取 true_val, 否则取 false_val.
        condition, true_val, false_val 都是 DataFrame
        """
        # 确保 condition 是布尔或数值
        cond = (condition > 0)
        return true_val.where(cond, false_val)

    @staticmethod
    def SignedPower(x, p):
        return x.abs() ** p * np.sign(x)

# 为了 eval() 方便，将算子放入字典
OP_DICT = {
    'Add': Operators.Add, 'Sub': Operators.Sub, 'Mul': Operators.Mul, 'Div': Operators.Div,
    'Neg': Operators.Neg, 'Abs': Operators.Abs, 'Log': Operators.Log, 'Sign': Operators.Sign,
    'TsRank': Operators.TsRank, 'CsRank': Operators.CsRank,
    'Delay': Operators.Delay, 'Delta': Operators.Delta,
    'Std': Operators.Std, 'Mean': Operators.Mean, 'Max': Operators.Max, 'Min': Operators.Min,
    'IfElse': Operators.IfElse, 'SignedPower': Operators.SignedPower
}

# ==========================================
# 4. 评估引擎 (Evaluation Layer)
# ==========================================

class FactorEngine:
    def __init__(self, data_dict):
        self.data = data_dict
        # 将基础数据放入 eval 上下文
        self.context = OP_DICT.copy()
        self.context.update({
            'open': data_dict['open'],
            'high': data_dict['high'],
            'low': data_dict['low'],
            'close': data_dict['close'],
            'volume': data_dict['volume'],
            'vwap': data_dict['vwap'],
            'returns': data_dict['returns']
        })

    def evaluate(self, formula):
        """
        计算因子值并评估 IC
        """
        try:
            # 1. 计算因子值 (执行公式)
            # 安全性提示: 在生产环境中不要直接用 eval，应使用 AST 解析
            factor_values = eval(formula, {"__builtins__": None}, self.context)

            # 2. 数据清洗 (去极值、标准化、填缺失)
            factor_values = factor_values.replace([np.inf, -np.inf], np.nan)

            # 3. 计算 IC (Information Coefficient)
            # IC = SpearmanCorr(Factor_t, Return_t+1)
            ic_series = []
            target = self.data['target']

            # 按时间截面计算相关性
            for date in factor_values.index:
                f_row = factor_values.loc[date]
                t_row = target.loc[date]

                # 移除 NaN
                valid_mask = ~(f_row.isna() | t_row.isna())
                if valid_mask.sum() > 2: # 至少要有几个股票才有意义
                    corr, _ = spearmanr(f_row[valid_mask], t_row[valid_mask])
                    if not np.isnan(corr):
                        ic_series.append(corr)

            if not ic_series:
                return None, 0.0

            mean_ic = np.mean(ic_series)
            # ICIR = mean / std (这里简化处理，只返回 IC)

            return factor_values, mean_ic

        except Exception as e:
            # print(f"评估失败: {formula}, 错误: {e}")
            return None, 0.0

    def calculate_correlation(self, factor_a, factor_b):
        """计算两个因子的截面相关性均值，用于去重"""
        corrs = []
        for date in factor_a.index:
            row_a = factor_a.loc[date]
            row_b = factor_b.loc[date]
            valid = ~(row_a.isna() | row_b.isna())
            if valid.sum() > 2:
                c, _ = spearmanr(row_a[valid], row_b[valid])
                corrs.append(c)
        return np.abs(np.mean(corrs)) if corrs else 1.0

# ==========================================
# 5. 经验记忆与 Agent (Memory & Agent Layer)
# ==========================================

class ExperienceMemory:
    def __init__(self):
        self.factor_library = []  # 存储已采纳的因子 {'formula': str, 'ic': float, 'values': df}
        self.rejected_history = [] # 存储失败经验

    def add_factor(self, formula, ic, values):
        self.factor_library.append({
            'formula': formula,
            'ic': ic,
            'values': values
        })

    def add_rejection(self, formula, reason):
        self.rejected_history.append(f"Formula: {formula} -> Rejected: {reason}")
        if len(self.rejected_history) > 10:
            self.rejected_history.pop(0) # 保持记忆简洁

    def get_prompt_context(self):
        """构建 Prompt 的上下文，包含成功的 Pattern 和失败的教训"""
        success_examples = [f['formula'] for f in sorted(self.factor_library, key=lambda x: abs(x['ic']), reverse=True)[:5]]

        context = "Existing successful factors (Alpha Library):\n"
        if not success_examples:
            context += "None yet.\n"
        else:
            for f in success_examples:
                context += f"- {f}\n"

        context += "\nRecent Rejections (Avoid these patterns):\n"
        if not self.rejected_history:
            context += "None yet.\n"
        else:
            for r in self.rejected_history[-5:]:
                context += f"- {r}\n"

        return context

class FactorMinerAgent:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("请设置 API_KEY！")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash') # 使用较快的模型

    def generate_factors(self, memory_context, n=5):
        """
        调用 LLM 生成因子公式
        """
        system_prompt = """
        You are an expert Quantitative Alpha Factor Researcher.
        Your goal is to discover valid stock market alpha factors using a specific Operator Library.

        The available data fields are: open, high, low, close, volume, vwap, returns.

        The available operators are:
        - Arithmetic: Add(x,y), Sub(x,y), Mul(x,y), Div(x,y), Neg(x), Abs(x), Log(x), SignedPower(x, p)
        - Time-Series: TsRank(x, window), Delay(x, window), Delta(x, window), Std(x, window), Mean(x, window), Max(x, window), Min(x, window)
        - Cross-Sectional: CsRank(x) (Returns values between 0 and 1)
        - Logic: IfElse(cond, true_val, false_val)

        Rules:
        1. Output ONLY a list of Python-like formulas.
        2. Do not use markdown code blocks.
        3. One formula per line.
        4. Focus on 'reversion' or 'momentum' logic.
        5. Try to find factors that are orthogonal (uncorrelated) to existing factors.
        6. Window sizes should be integers like 5, 10, 20.

        Example format:
        TsRank(close, 10)
        Neg(CsRank(Delta(volume, 5)))
        """

        user_prompt = f"""
        {memory_context}

        Task: Generate {n} NEW alpha factor formulas.
        Try to combine operators creatively.
        Avoid the rejected patterns.
        Maximize Information Coefficient (IC).
        """

        try:
            response = self.model.generate_content(system_prompt + "\n" + user_prompt)
            text = response.text
            # 简单的解析：提取非空行
            formulas = [line.strip() for line in text.split('\n') if line.strip() and '(' in line]
            return formulas
        except Exception as e:
            print(f"LLM Call Failed: {e}")
            return []

# ==========================================
# 6. 主程序 (The Ralph Loop)
# ==========================================

def main():
    if not API_KEY:
        print("错误: 未检测到 API_KEY。请在代码中配置您的 Google Gemini API Key。")
        return

    # 1. 初始化
    print(">>> 正在初始化 FactorMiner...")
    market_data = get_real_data(TICKERS, START_DATE, END_DATE)
    engine = FactorEngine(market_data)
    memory = ExperienceMemory()
    agent = FactorMinerAgent(API_KEY)

    # 2. Ralph Loop 循环
    for iteration in range(ITERATIONS):
        print(f"\n{'='*20} Iteration {iteration + 1} / {ITERATIONS} {'='*20}")

        # --- Retrieve (检索) ---
        context = memory.get_prompt_context()
        print(f"Memory Context Retrieved. Library Size: {len(memory.factor_library)}")

        # --- Generate (生成) ---
        print("Agent is brainstorming factors...")
        candidates = agent.generate_factors(context, n=FACTORS_PER_BATCH)
        print(f"Generated {len(candidates)} candidates.")

        # --- Evaluate (评估) & Distill (提炼) ---
        for formula in candidates:
            print(f"  Validating: {formula[:60]}...", end="")

            # Stage 1: Check Syntax & Calculate IC
            f_values, ic = engine.evaluate(formula)

            if f_values is None:
                print(" -> [Syntax Error / Runtime Error]")
                memory.add_rejection(formula, "Execution Failed")
                continue

            if abs(ic) < IC_THRESHOLD:
                print(f" -> [Rejected] Low IC: {ic:.4f}")
                memory.add_rejection(formula, f"Low IC ({ic:.4f})")
                continue

            # Stage 2: Check Correlation (Redundancy)
            is_redundant = False
            for existing in memory.factor_library:
                corr = engine.calculate_correlation(f_values, existing['values'])
                if corr > CORR_THRESHOLD:
                    print(f" -> [Rejected] High Corr ({corr:.2f}) with existing factor")
                    memory.add_rejection(formula, f"High Correlation with {existing['formula'][:20]}...")
                    is_redundant = True
                    break

            if is_redundant:
                continue

            # Stage 3: Admit to Library
            print(f" -> [Admitted!] IC: {ic:.4f}")
            memory.add_factor(formula, ic, f_values)

    # ==========================================
    # 7. 结果展示
    # ==========================================
    print(f"\n\n{'='*20} Final Factor Library {'='*20}")
    print(f"Total Factors Discoverd: {len(memory.factor_library)}")

    result_df = pd.DataFrame([
        {'Formula': f['formula'], 'IC': f['ic']}
        for f in memory.factor_library
    ])

    if not result_df.empty:
        result_df = result_df.sort_values(by='IC', key=abs, ascending=False).reset_index(drop=True)
        print(result_df)
        result_df.to_csv("discovered_factors.csv")
        print("\n因子已保存至 discovered_factors.csv")
    else:
        print("未发现有效因子。尝试增加迭代次数或调整阈值。")

if __name__ == "__main__":
    main()
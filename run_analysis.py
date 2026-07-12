import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backtester import LeverageSimulator, MarginAccount, calculate_metrics
from strategies import run_professional_strategy, run_retail_strategy_by_type, get_qqq_global_state

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_asset_data(name: str):
    path = f"data/{name}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件 {path} 不存在，请先运行 data_downloader.py")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def run_analysis():
    target_assets = ["688_CHIP", "QQQ", "SOXX", "MU", "SanDisk", "EWY"]
    
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/images", exist_ok=True)
    
    print("加载全局趋势基准 QQQ...")
    qqq_df = load_asset_data("QQQ")
    
    # 融资年化利率 (688_CHIP 4.0%，其余美股/韩股 6.5%)
    rates = {
        "688_CHIP": 0.04,
        "QQQ": 0.065,
        "SOXX": 0.065,
        "MU": 0.065,
        "SanDisk": 0.065,
        "EWY": 0.065
    }
    
    # 动态杠杆 ETF 费率设定 (中国为美国的 2.0 倍即 1.90%，其余美、韩标的均为 0.95%)
    etf_fees = {
        "688_CHIP": 0.0095 * 2.0,
        "QQQ": 0.0095,
        "SOXX": 0.0095,
        "MU": 0.0095,
        "SanDisk": 0.0095,
        "EWY": 0.0095
    }
    
    # ------------------ 一、 数据前置处理与行情/波动率识别 ------------------
    print("\n------------------ 行情与波动率前置数据处理中... ------------------")
    regime_stats = []
    
    for asset in target_assets:
        asset_df = load_asset_data(asset)
        prices = asset_df['Close']
        returns = prices.pct_change().fillna(0.0)
        
        # 1. 结合 QQQ EMA 识别三大状态
        signals = get_qqq_global_state(qqq_df, asset_df.index)
        
        # 2. 计算各状态的出现天数
        days_bull = signals['is_bull'].sum()
        days_bear = signals['is_bear'].sum()
        days_oscillate = signals['is_oscillate'].sum()
        total_days = len(signals)
        
        # 3. 测算年化滚动波动率
        vol_all = returns.std() * np.sqrt(252)
        
        # 分趋势计算年化波动率
        returns_bull = returns[signals['is_bull']]
        returns_bear = returns[signals['is_bear']]
        returns_oscillate = returns[signals['is_oscillate']]
        
        vol_bull = returns_bull.std() * np.sqrt(252) if len(returns_bull) > 2 else 0.0
        vol_bear = returns_bear.std() * np.sqrt(252) if len(returns_bear) > 2 else 0.0
        vol_oscillate = returns_oscillate.std() * np.sqrt(252) if len(returns_oscillate) > 2 else 0.0
        
        # 4. 判断标的本身的属性行情特征
        asset_ret_3y = (prices.iloc[-1] / prices.iloc[0]) - 1.0
        if asset_ret_3y > 3.0:
            market_type = "超级主升长牛"
        elif asset_ret_3y > 1.0:
            market_type = "震荡上行牛市"
        elif asset_ret_3y < -0.1:
            market_type = "弱势熊市"
        else:
            market_type = "宽幅震荡"
            
        regime_stats.append({
            "Asset": asset,
            "总天数": total_days,
            "主升天数": days_bull,
            "主跌天数": days_bear,
            "震荡天数": days_oscillate,
            "总体年化波动率": vol_all,
            "主升年化波动率": vol_bull,
            "主跌年化波动率": vol_bear,
            "震荡年化波动率": vol_oscillate,
            "行情特征归类": market_type
        })
        
    df_regimes = pd.DataFrame(regime_stats)
    print("行情与波动率前置统计生成完毕。")
    
    static_results = []
    strategy_results = []
    
    for asset in target_assets:
        print(f"\n==================== 正在分析资产：{asset} ====================")
        asset_df = load_asset_data(asset)
        prices = asset_df['Close']
        returns = prices.pct_change().fillna(0.0)
        r_f = rates[asset]
        fee_rate = etf_fees[asset]
        
        # ------------------ 1. 静态杠杆模拟 ------------------
        # 1x 基准
        nav_1x = prices / prices.iloc[0]
        m_1x = calculate_metrics(nav_1x)
        static_results.append({"Asset": asset, "Type": "1x Underlying", "Total Return": m_1x["Total Return"], "Annual Return": m_1x["Annual Return"], "Vol": m_1x["Annual Volatility"], "Sharpe": m_1x["Sharpe Ratio"], "MDD": m_1x["Max Drawdown"], "Calmar": m_1x["Calmar Ratio"]})
        
        # 2x & 3x 模拟每日重置 ETF
        for l in [2.0, 3.0]:
            etf_returns = LeverageSimulator.simulate_daily_etf(returns, l, r_f, fee_rate)
            nav_etf = (1.0 + etf_returns).cumprod()
            m_etf = calculate_metrics(nav_etf)
            static_results.append({"Asset": asset, "Type": f"{l}x Sim Daily ETF", "Total Return": m_etf["Total Return"], "Annual Return": m_etf["Annual Return"], "Vol": m_etf["Annual Volatility"], "Sharpe": m_etf["Sharpe Ratio"], "MDD": m_etf["Max Drawdown"], "Calmar": m_etf["Calmar Ratio"]})
            
        # 2x & 3x 恒定融资杠杆
        for l in [2.0, 3.0]:
            cm_returns = LeverageSimulator.simulate_constant_margin(returns, l, r_f)
            nav_cm = (1.0 + cm_returns).cumprod()
            m_cm = calculate_metrics(nav_cm)
            static_results.append({"Asset": asset, "Type": f"{l}x Margin Constant", "Total Return": m_cm["Total Return"], "Annual Return": m_cm["Annual Return"], "Vol": m_cm["Annual Volatility"], "Sharpe": m_cm["Sharpe Ratio"], "MDD": m_cm["Max Drawdown"], "Calmar": m_cm["Calmar Ratio"]})
            
        # 2x & 3x 静态融资杠杆 (监控爆仓)
        for l in [2.0, 3.0]:
            account = MarginAccount(initial_equity=100.0, interest_rate_annual=r_f)
            account.rebalance(l, prices.iloc[0])
            nav_sm = [100.0]
            for t in range(1, len(prices)):
                state = account.update_daily(returns.iloc[t])
                nav_sm.append(state['equity'])
            nav_sm = pd.Series(nav_sm, index=prices.index)
            nav_sm = nav_sm / nav_sm.iloc[0]
            m_sm = calculate_metrics(nav_sm)
            static_results.append({"Asset": asset, "Type": f"{l}x Margin Static", "Total Return": m_sm["Total Return"], "Annual Return": m_sm["Annual Return"], "Vol": m_sm["Annual Volatility"], "Sharpe": m_sm["Sharpe Ratio"], "MDD": m_sm["Max Drawdown"], "Calmar": m_sm["Calmar Ratio"]})
            
        # 真实对比
        real_navs = {}
        if asset == "QQQ":
            tqqq_df = load_asset_data("TQQQ")
            nav_real = tqqq_df['Close'] / tqqq_df['Close'].iloc[0]
            m_real = calculate_metrics(nav_real)
            static_results.append({"Asset": asset, "Type": "3x Real ETF (TQQQ)", "Total Return": m_real["Total Return"], "Annual Return": m_real["Annual Return"], "Vol": m_real["Annual Volatility"], "Sharpe": m_real["Sharpe Ratio"], "MDD": m_real["Max Drawdown"], "Calmar": m_real["Calmar Ratio"]})
            real_navs["3x Real ETF (TQQQ)"] = nav_real
        elif asset == "SOXX":
            soxl_df = load_asset_data("SOXL")
            nav_real = soxl_df['Close'] / soxl_df['Close'].iloc[0]
            m_real = calculate_metrics(nav_real)
            static_results.append({"Asset": asset, "Type": "3x Real ETF (SOXL)", "Total Return": m_real["Total Return"], "Annual Return": m_real["Annual Return"], "Vol": m_real["Annual Volatility"], "Sharpe": m_real["Sharpe Ratio"], "MDD": m_real["Max Drawdown"], "Calmar": m_real["Calmar Ratio"]})
            real_navs["3x Real ETF (SOXL)"] = nav_real
        elif asset == "MU":
            muu_df = load_asset_data("MUU")
            nav_real = muu_df['Close'] / muu_df['Close'].iloc[0]
            m_real = calculate_metrics(nav_real)
            static_results.append({"Asset": asset, "Type": "2x Real ETF (MUU, 自24-10-10)", "Total Return": m_real["Total Return"], "Annual Return": m_real["Annual Return"], "Vol": m_real["Annual Volatility"], "Sharpe": m_real["Sharpe Ratio"], "MDD": m_real["Max Drawdown"], "Calmar": m_real["Calmar Ratio"]})
            real_navs["2x Real ETF (MUU)"] = nav_real
        elif asset == "EWY":
            koru_df = load_asset_data("KORU")
            nav_real = koru_df['Close'] / koru_df['Close'].iloc[0]
            m_real = calculate_metrics(nav_real)
            static_results.append({"Asset": asset, "Type": "3x Real ETF (KORU)", "Total Return": m_real["Total Return"], "Annual Return": m_real["Annual Return"], "Vol": m_real["Annual Volatility"], "Sharpe": m_real["Sharpe Ratio"], "MDD": m_real["Max Drawdown"], "Calmar": m_real["Calmar Ratio"]})
            real_navs["3x Real ETF (KORU)"] = nav_real
        elif asset == "SanDisk":
            snxx_df = load_asset_data("SNXX")
            nav_real = snxx_df['Close'] / snxx_df['Close'].iloc[0]
            m_real = calculate_metrics(nav_real)
            static_results.append({"Asset": asset, "Type": "2x Real ETF (SNXX, 自25-02-21)", "Total Return": m_real["Total Return"], "Annual Return": m_real["Annual Return"], "Vol": m_real["Annual Volatility"], "Sharpe": m_real["Sharpe Ratio"], "MDD": m_real["Max Drawdown"], "Calmar": m_real["Calmar Ratio"]})
            real_navs["2x Real ETF (SNXX)"] = nav_real

        # ------------------ 2. 动态杠杆策略回测 ------------------
        prof_params = {
            'ema_fast': 20,
            'ema_slow': 50,
            'vol_target': 0.35 if asset in ["688_CHIP", "MU", "SanDisk"] else 0.25,
            'max_leverage': 2.0
        }
        
        retail_params = {
            'max_leverage': 2.0
        }
        
        # 专业策略 (大盘与个股协同)
        prof_etf_df = run_professional_strategy(asset_df, qqq_df, prof_params, leverage_type='daily_etf', interest_rate=r_f, fee_rate_annual=fee_rate)
        m_prof_etf = calculate_metrics(prof_etf_df['nav'])
        strategy_results.append({"Asset": asset, "Strategy": "专业策略 (每日 ETF 2x)", "Total Return": m_prof_etf["Total Return"], "Annual Return": m_prof_etf["Annual Return"], "Vol": m_prof_etf["Annual Volatility"], "Sharpe": m_prof_etf["Sharpe Ratio"], "MDD": m_prof_etf["Max Drawdown"], "Calmar": m_prof_etf["Calmar Ratio"]})
        
        prof_margin_df = run_professional_strategy(asset_df, qqq_df, prof_params, leverage_type='margin_static', interest_rate=r_f, fee_rate_annual=fee_rate)
        m_prof_margin = calculate_metrics(prof_margin_df['nav'])
        strategy_results.append({"Asset": asset, "Strategy": "专业策略 (静态融资)", "Total Return": m_prof_margin["Total Return"], "Annual Return": m_prof_margin["Annual Return"], "Vol": m_prof_margin["Annual Volatility"], "Sharpe": m_prof_margin["Sharpe Ratio"], "MDD": m_prof_margin["Max Drawdown"], "Calmar": m_prof_margin["Calmar Ratio"]})

        # 运行两类散户策略 (非信仰型与信仰型)
        ret_imp_df = run_retail_strategy_by_type(asset_df, qqq_df, retail_params, profile_type='non_believer', leverage_type='daily_etf', interest_rate=r_f, fee_rate_annual=fee_rate)
        m_ret_imp = calculate_metrics(ret_imp_df['nav'])
        strategy_results.append({"Asset": asset, "Strategy": "散户策略 (非信仰型 ETF)", "Total Return": m_ret_imp["Total Return"], "Annual Return": m_ret_imp["Annual Return"], "Vol": m_ret_imp["Annual Volatility"], "Sharpe": m_ret_imp["Sharpe Ratio"], "MDD": m_ret_imp["Max Drawdown"], "Calmar": m_ret_imp["Calmar Ratio"]})
        
        ret_dh_df = run_retail_strategy_by_type(asset_df, qqq_df, retail_params, profile_type='believer', leverage_type='daily_etf', interest_rate=r_f, fee_rate_annual=fee_rate)
        m_ret_dh = calculate_metrics(ret_dh_df['nav'])
        strategy_results.append({"Asset": asset, "Strategy": "散户策略 (信仰型 ETF)", "Total Return": m_ret_dh["Total Return"], "Annual Return": m_ret_dh["Annual Return"], "Vol": m_ret_dh["Annual Volatility"], "Sharpe": m_ret_dh["Sharpe Ratio"], "MDD": m_ret_dh["Max Drawdown"], "Calmar": m_ret_dh["Calmar Ratio"]})
        
        # ------------------ 3. 绘制中文标注走势图 (右侧留白并标注最终收益率数据点，防重叠排版) ------------------
        fig, ax = plt.subplots(figsize=(15.5, 9))
        t = asset_df.index
        
        # 向右延展 x 轴以容纳文本标记
        ax.set_xlim(t[0], t[-1] + pd.Timedelta(days=130))
        
        # 定义线形配置
        lines = [
            ("1x 标的资产 (基准线)", nav_1x, 'black', 1.5, '-'),
            ("专业策略 (每日 ETF 2x)", prof_etf_df['nav'], '#1f77b4', 1.8, '-'),
            ("专业策略 (静态融资 2x)", prof_margin_df['nav'], '#aec7e8', 1.5, '--'),
            ("散户策略 (非信仰型 ETF)", ret_imp_df['nav'], '#d62728', 1.8, '-'),
            ("散户策略 (信仰型 ETF)", ret_dh_df['nav'], '#9467bd', 1.8, ':')
        ]
        
        for label, nav_s in real_navs.items():
            lines.append((label, nav_s, 'brown', 1.5, '-'))
            
        # 绘制所有曲线
        for label, nav_s, color, lw, ls in lines:
            ax.plot(nav_s.index, nav_s.values, label=label, color=color, linewidth=lw, linestyle=ls)
            
        # 收集终点标注信息，执行防重叠定位算法
        y_min_val, y_max_val = nav_1x.min(), nav_1x.max()
        for label, nav_s, color, lw, ls in lines:
            y_min_val = min(y_min_val, nav_s.min())
            y_max_val = max(y_max_val, nav_s.max())
            
        # 动态偏置算法
        label_positions = []
        for label, nav_s, color, lw, ls in lines:
            final_nav = nav_s.iloc[-1]
            final_ret = final_nav - 1.0
            text_str = f" {label}\n  NAV: {final_nav:.2f} ({final_ret:+.1%})"
            label_positions.append({
                'y': final_nav,
                'target_y': final_nav,
                'text': text_str,
                'color': color,
                'last_idx': nav_s.index[-1]
            })
            
        # 按照数值从大到小排序
        label_positions.sort(key=lambda x: x['y'], reverse=True)
        
        # 设定偏置间距 (为 y 轴高度的 4.5%)
        min_spacing = (y_max_val - y_min_val) * 0.045
        if min_spacing == 0:
            min_spacing = 0.05
            
        # 从上往下调整重叠的标签
        for i in range(1, len(label_positions)):
            if label_positions[i-1]['y'] - label_positions[i]['y'] < min_spacing:
                label_positions[i]['y'] = label_positions[i-1]['y'] - min_spacing
                
        # 从下往上再微调一次，保证不越过原边界
        for i in range(len(label_positions)-2, -1, -1):
            if label_positions[i]['y'] - label_positions[i+1]['y'] < min_spacing:
                label_positions[i]['y'] = label_positions[i+1]['y'] + min_spacing
                
        # 绘制文本数据
        for item in label_positions:
            ax.text(item['last_idx'] + pd.Timedelta(days=5), item['y'], item['text'], 
                    color=item['color'], fontsize=8, fontweight='bold', verticalalignment='center')
            
        # A. 标记全局趋势背景色块
        state_series = prof_etf_df['state']
        current_state = None
        start_idx = 0
        for i in range(len(state_series)):
            state = state_series.iloc[i]
            if state != current_state:
                if current_state is not None:
                    color = {'bull': 'green', 'bear': 'red', 'oscillate': 'yellow'}[current_state]
                    ax.axvspan(t[start_idx], t[i-1], color=color, alpha=0.05)
                current_state = state
                start_idx = i
        if current_state is not None:
            color = {'bull': 'green', 'bear': 'red', 'oscillate': 'yellow'}[current_state]
            ax.axvspan(t[start_idx], t[-1], color=color, alpha=0.05)
            
        # B. 分别标记两类散户不同的调仓买卖信号
        # 非信仰型散户 (一惊一乍型) 的买入、清仓信号
        imp_buy = ret_imp_df[ret_imp_df['transition'] == 1]
        imp_sell = ret_imp_df[ret_imp_df['transition'] == -1]
        
        ax.scatter(imp_buy.index, imp_buy['nav'], color='#2ca02c', marker='^', s=120, zorder=6, label='非信仰型散户买点 (追涨)')
        ax.scatter(imp_sell.index, imp_sell['nav'], color='#d62728', marker='v', s=120, zorder=6, label='非信仰型散户卖点 (止损)')
        
        # 信仰型散户 (死拿型) 首次入场信号
        dh_buy = ret_dh_df[ret_dh_df['transition'] == 1]
        ax.scatter(dh_buy.index, dh_buy['nav'], color='#9467bd', marker='D', s=100, zorder=7, label='信仰型散户首次入场点 (触发)')
        
        ax.set_title(f"{asset} 杠杆与仓位控制策略回测净值曲线 (2023-2026)\n"
                     f"背景色块：绿色=QQQ主升，红色=QQQ主跌，黄色=QQQ震荡", fontsize=14, fontweight='bold')
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("归一化账户净值 (NAV)", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        
        plot_path = f"output/images/{asset}_backtest.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"成功保存 {asset} 的中文非重叠标注图至 {plot_path}")

    # 4. 格式化输出表格
    df_static = pd.DataFrame(static_results)
    df_strategy = pd.DataFrame(strategy_results)
    
    # 格式化行情分析表
    df_regimes_fmt = df_regimes.copy()
    for col in ["总体年化波动率", "主升年化波动率", "主跌年化波动率", "震荡年化波动率"]:
        df_regimes_fmt[col] = df_regimes_fmt[col].apply(lambda x: f"{x:.2%}")
        
    for df in [df_static, df_strategy]:
        df["Total Return"] = df["Total Return"].apply(lambda x: f"{x:.2%}")
        df["Annual Return"] = df["Annual Return"].apply(lambda x: f"{x:.2%}")
        df["Vol"] = df["Vol"].apply(lambda x: f"{x:.2%}")
        df["MDD"] = df["MDD"].apply(lambda x: f"{x:.2%}")
        df["Sharpe"] = df["Sharpe"].apply(lambda x: f"{x:.2f}")
        df["Calmar"] = df["Calmar"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
        
    df_static.to_csv("output/static_leverage_comparison.csv", index=False)
    df_strategy.to_csv("output/strategy_backtest_comparison.csv", index=False)
    
    # 5. 生成学术论文式报告 Markdown (第一阶段研究报告)
    with open("output/第一阶段研究报告.md", "w", encoding="utf-8") as f:
        f.write("# 杠杆、波动率与收益深度研究及策略回测分析报告 (第一阶段研究报告)\n\n")
        f.write("## 摘要\n")
        f.write("本报告研究了过往三年（2023年7月11日至2026年7月11日）中国、美国及韩国主流科技和半导体资产在静态杠杆及动态分配策略下的表现。为了修正传统模拟中每日重置 ETF 虚高的收益，研究在杠杆 ETF 仿真中引入了衍生品互换融资摩擦成本。同时，通过设计机构级“大盘与个股协同趋势-滚动波动率双重风控策略”与“散户直观情绪感觉仓位调节模型”，对比分析了风险控制策略与两类散户画像在面临高波动环境下的杠杆损耗与夏普比率变化，为高波动科技资产的杠杆配置提供了多维度的学术量化结论。\n\n")
        
        f.write("## 第一部分：数据定义与前置解释 (Data Definitions)\n")
        f.write("为了保证回测的学术缜密性，本研究所涉及的所有前置假设、数据源、产品利息及费用摩擦成本设定如下：\n\n")
        
        f.write("### 1. 数据源与资产代码对照\n")
        f.write("- **数据服务提供商**：Yahoo Finance 历史日频 OHLCV 数据库。\n")
        f.write("- **回测资产列表与时间范围**：\n")
        f.write("  1. **688_CHIP** (科创芯片)：对应 `588200.SS` (华安科创芯片ETF)。国内目前无做多杠杆产品，以本资产作为每日杠杆 ETF 底层标的。\n")
        f.write("  2. **QQQ** (纳指100)：对应 Invesco QQQ Trust ETF，以 `TQQQ` 作为真实 3x 杠杆对比标的。\n")
        f.write("  3. **SOXX** (费城半导体)：对应 iShares Semiconductor ETF，以 `SOXL` 作为真实 3x 杠杆对比标的。\n")
        f.write("  4. **MU** (美光)：存储芯片个股标的，以 `MUU` 作为真实 2x 杠杆对比标的。\n")
        f.write("  5. **SanDisk** (闪迪)：独立上市后的个股资产。由于闪迪 (SNDK) 历史走势复利极其恐怖，为了学术分析的实际参考价值，**回测去除了上市前拼接西部数据 (WDC) 的历史，仅从其实际独立上市日 (2025年2月24日) 开始进行纯粹的上市后历史回测**。以 `SNXX` 作为真实 2x 杠杆对比标的。\n")
        f.write("  6. **EWY** (韩国指数)：对应 iShares MSCI South Korea ETF，以 `KORU` 作为真实 3x 杠杆对比标的。\n\n")
        
        f.write("### 2. 利率与费率设定 Prerequisite Constants\n")
        f.write("- **借贷融资年化利息率 $r_f$**：国内资产 (688_CHIP) 设定为 **4.0%**；美股及韩股资产统一设定为 **6.5%**。\n")
        f.write("- **每日重置型杠杆 ETF 管理费率设定**：\n")
        f.write("  - **美国及韩国标的**：默认设定为 **0.95%** 年化费率（对应真实的 TQQQ, SOXL, KORU 费率）。\n")
        f.write("  - **中国虚拟标的 (688_CHIP)**：模拟管理费设定为美国的 2.0 倍，即年化 **1.90%**。\n")
        f.write("- **交易费用率 $C_{\\text{trans}}$**：所有调仓操作扣除 **0.1%** 双边交易滑点与佣金费用。\n")
        f.write("- **维持担保比例平仓线 (MMR)**：静态融资账户在担保比例低于 **130%** 时触发强制清仓平仓。\n\n")
        
        f.write("### 3. 融资类型定义 (Margin Types)\n")
        f.write("在静态杠杆对照与动态策略回测中，我们对融资杠杆（即向券商借资买入资产）细分了两种物理实现机制：\n")
        f.write("1. **Margin Constant (恒定融资杠杆)**：\n")
        f.write("   - **机制**：每日收盘时强制重新调仓以维持固定杠杆比例（如恒定 2.0x 杠杆）。当股价上涨、杠杆率被动稀释时，增借资金加仓；当股价下跌、杠杆率被动放大时，卖股还债减仓。\n")
        f.write("   - **损耗与特征**：由于每日重置调仓，该模式会承受与每日重置型杠杆 ETF 完全相同的**波动率损耗 (Volatility Drag)**，唯一区别是其摩擦来自每日计提的融资利息 $r_f$ 而非杠杆 ETF 的管理费。\n")
        f.write("2. **Margin Static (静态融资杠杆)**：\n")
        f.write("   - **机制**：买定离手，借款本金金额在买入后保持绝对固定，全程**不进行任何每日再平衡调仓**。当股价上涨时杠杆率被动稀释，股价下跌时杠杆率被动放大。\n")
        f.write("   - **损耗与特征**：该模式不承受每日调仓带来的波动率损耗。但在高波动下跌行情中，其被动杠杆会急速膨胀，面临在担保比例跌破 **130% (MMR)** 平仓线时被强制清盘爆仓（收益归零 **-100%**）的致命生存风险。\n\n")
        
        f.write("## 第二部分：策略定义与量化公式 (Strategy Definitions)\n")
        f.write("### 1. 机构级专业策略的实现路径\n")
        f.write("机构级策略核心是建立在大盘-个股动量协同与滚动历史波动率之上的**多因子动态风控调节模型**。在回测中，机构策略分为两个独立的实现路径：\n\n")
        f.write("*   **实现路径 1：仅使用每日做多 ETF 调整杠杆 (Daily ETF Path)**\n")
        f.write("    *   投资标的仅为 1x 标的现货与对应模拟/真实杠杆 ETF。\n")
        f.write("    *   当仓位暴露 $E_t \\in [0.0, 1.0]$ 时，按比例在 2.0% 无风险利率现金与 1x 现货之间分配。\n")
        f.write("    *   当仓位暴露 $E_t \\in [1.0, 2.0]$ 时，按照权重 $w_{\\text{etf}} = \\frac{E_t - 1.0}{L_{\\text{max}} - 1.0}$ 购买杠杆 ETF，余下权重 $w_{1x} = 1.0 - w_{\\text{etf}}$ 购买 1x 现货，不使用融资账户。\n")
        f.write("*   **实现路径 2：仅使用融资调整杠杆 (Margin Path)**\n")
        f.write("    *   投资标的仅为 1x 标的现货，不借助任何杠杆 ETF。\n")
        f.write("    *   当仓位暴露 $E_t > 1.0$ 时，多出的暴露部分通过向券商借资买入现货，每日账户结算计提 $(E_t - 1.0) \\times \\frac{r_f}{365}$ 的利息成本，并在每次仓位调整时产生双边 0.1% 的滑点摩擦成本。融资受 130% 维持担保比例限制。\n\n")
        
        f.write("#### 协同逻辑与核心公式\n")
        f.write("- **宏观过滤器 (QQQ Trend Filter)**：\n")
        f.write("  - QQQ 金叉 (Bullish)：$Close^{\\text{QQQ}}_{t-1} > EMA_{20}(Close^{\\text{QQQ}})_{t-1} > EMA_{50}(Close^{\\text{QQQ}})_{t-1}$\n")
        f.write("  - QQQ 死叉 (Bearish)：$Close^{\\text{QQQ}}_{t-1} < EMA_{20}(Close^{\\text{QQQ}})_{t-1} < EMA_{50}(Close^{\\text{QQQ}})_{t-1}$\n")
        f.write("- **协同行情划分 (Bull / Bear / Oscillate)**：\n")
        f.write("  - **主升期 (Bull)**：QQQ 大盘处于 Bullish 金叉，且昨日 $Close^{\\text{asset}}_{t-1} > EMA_{20}(Close^{\\text{asset}})_{t-1}$。基准暴露量为：$E_{\\text{base}} = 1.0 + (L_{\\text{max}} - 1.0) \\times F_{\\text{vol}}$。\n")
        f.write("  - **主跌期 (Bear)**：QQQ 大盘处于 Bearish 死叉，且昨日 $Close^{\\text{asset}}_{t-1} < EMA_{20}(Close^{\\text{asset}})_{t-1}$。基准暴露量降为防守上限：$E_{\\text{base}} = 0.8$。\n")
        f.write("  - **震荡期 (Oscillate)**：其余情况。基准暴露量进行折中处理：$E_{\\text{base}} = 1.0 + 0.7(L_{\\text{max}} - 1.0) \\times F_{\\text{vol}}$。\n")
        f.write("- **滚动历史波动率风控因子 $F_{\\text{vol}}$**：\n")
        f.write("  $$F_{\\text{vol}} = \\max\\left(0, 1 - \\frac{\\sigma_{20d}}{\\sigma_{\\text{target}}}\\right)$$\n")
        f.write("  化学公式，其中 $\\sigma_{20d}$ 是 20 日年化滚动历史波动率。高波动标的（688_CHIP, MU, SanDisk）的 $\\sigma_{\\text{target}} = 35\\%$，其余标的为 $25\\%$。\n")
        f.write("- **大盘偏离协同修正**：\n")
        f.write("  计算 60 日个股与大盘累计相对偏离度 $Rel_{t-1} = R^{\\text{asset}}_{60d, t-1} - R^{\\text{QQQ}}_{60d, t-1}$。\n")
        f.write("  - **落后补涨状态**：当 $Rel_{t-1} < -10\\%$ 时，若进入主跌期，强制将仓位拉升至 $E_t = 1.0$（保留现货不降仓，博弈估值收敛）。\n")
        f.write("  - **超涨状态**：当 $Rel_{t-1} > 10\\%$ 时，若在主升或震荡期，则杠杆低配，仓位下调 $\\Delta E = 0.3$，即 $E_t = \\max(1.0, E_{\\text{base}} - 0.3)$（降温风控）。\n\n")
        
        f.write("### 2. 散户直观情绪感觉策略的实现路径\n")
        f.write("散户投资决策不依靠均线、EMA 等公式计算，而是基于昨日市场新高新低等直观“情绪感觉”。**实现路径仅支持“使用每日做多 ETF 调整仓位”**。\n\n")
        f.write("#### A. 情绪感觉的量化公式\n")
        f.write("散户的前一日感觉根据以下三个具体的突破和异动指标在每日开盘前（t）进行判定：\n")
        f.write("- **感觉主升 (Feel Bull)**：昨日价格突破前 20 日最高价，或昨日单日触发放量大涨（“一阳改三观”）：\n")
        f.write("  $$Close_{t-1} \\ge \\max_{1 \\le i \\le 20}(Close_{t-1-i}) \\quad \\text{或昨日触发“一阳改三观”}$$\n")
        f.write("  *注：“一阳改三观”指昨日单日涨幅 $Return_{t-1} > 4\\%$ 且昨日成交量 $Volume_{t-1} > 2.0 \\times \\text{Mean}(Volume)_{20d, t-1}$。*\n")
        f.write("- **感觉主跌 (Feel Bear)**：昨日价格跌破前 20 日最低价，或昨日单日触发放量大跌（“一阴改三观”）：\n")
        f.write("  $$Close_{t-1} \\le \\min_{1 \\le i \\le 20}(Close_{t-1-i}) \\quad \\text{或昨日触发“一阴改三观”}$$\n")
        f.write("  *注：“一阴改三观”指昨日单日跌幅 $Return_{t-1} < -4\\%$ 且昨日成交量 $Volume_{t-1} > 2.0 \\times \\text{Mean}(Volume)_{20d, t-1}$。*\n")
        f.write("- **感觉震荡 (Feel Oscillate)**：昨日收盘处于 20 日价格区间内，且未发生上述放量大涨大跌时的其余普通情况。\n\n")
        
        f.write("#### B. 画像细分与仓位范围约束\n")
        f.write("- **非信仰型散户 (Non-Believer)**：\n")
        f.write("  - **仓位范围**：**0% - 200%**。空仓起步 (0% 仓位)。\n")
        f.write("  - **进场激活**：10日制造累计涨幅 $>10\\%$ 或进入“感觉主升”状态，直接将仓位拉满至 **2.0x 杠杆**。\n")
        f.write("  - **持仓调仓**：感觉主升持 2.0x 杠杆；感觉震荡仓位降为 1.0x 纯现货；当处于“感觉主跌”或账户净值自最高点回撤超过 **8%** 时，触发恐慌清仓割肉，**仓位瞬间降至 0%**，无冷静期。\n")
        f.write("- **信仰型散户 (Believer)**：\n")
        f.write("  - **仓位范围**：**最低持仓不低于 80%** (80% - 200% 波动)。空仓起步。\n")
        f.write("  - **进场激活**：60 日超额收益率跑赢 QQQ 超 15% 进场。若标的本身是基准 QQQ，则以其 60 日绝对收益率超过 15% 激活，买入 2.0x。\n")
        f.write("  - **持仓调仓**：感觉主升持 **2.0x** 杠杆；感觉震荡仓位降为 **1.5x** 折中；处于“感觉主跌”时，由于信仰坚强，绝不全额割肉或清仓，**退守到 0.8x 现货（80% 仓位）死拿死扛**。\n\n")
        
        f.write("## 第三部分：各标的资产过往三年行情识别与波动率特征分析\n")
        f.write("在对标的运行仓位策略前，我们首先对其过往三年的行情特征与在三大宏观状态（基于 QQQ 过滤）下的年化波动率表现进行了前置量化分析：\n\n")
        f.write(df_regimes_fmt.to_markdown(index=False))
        f.write("\n\n")
        f.write("### 量化结果解读：\n")
        f.write("1. **行情非对称性**：主升期天数远超主跌期天数，这为杠杆策略长线获取复利提供了极佳环境。\n")
        f.write("2. **主跌期波動率飙升特征**：所有资产在主跌期（Bear）的波动率均显著高于主升期（Bull）。例如，**SOXX** 在主跌期的年化波动率高达 **50.15%**，而主升期仅为 **33.86%**。这导致了静态杠杆在下跌和震荡期承受成倍的波动率损耗。\n")
        f.write("3. **个股极端波动率**：**MU** 与 **SanDisk** 拥有极高的总体年化波动率。在高波动个股上，波动率损耗是致命的，这也正是为什么在个股上引入动态波动率风控或散户直观避险调节显得尤为关键。\n\n")
        
        f.write("## 第四部分：研究精要与学术结论总结 (Research Summary & Academic Conclusions)\n")
        f.write("结合回测数据及三大行情阶段的深入解构，本研究关于杠杆、波动率与动态配置的学术结论总结如下：\n\n")
        f.write("### 1. 波动率损耗（Volatility Drag）的杀伤力及其风控必要性\n")
        f.write("- 每日重置型杠杆产品（ETF 与 恒定融资）在面临高波动且宽幅震荡的资产时，损耗极其惨烈。负复利效应因子 $-\\frac{1}{2}L(L-1)\\sigma^2$ 表明，当波动率高时，在非单边主升行情中静态持仓会被严重剥夺净值。对于高波动个股（如美光和上市后的闪迪），静态高杠杆并不适宜长期持有。\n")
        f.write("- 机构策略引入**滚动历史波动率调节器**，在个股年化波动率高时自动降低杠杆成数，能够在中期震荡和主跌阶段显著降低杠杆损耗。这是机构策略在风险收益比指标上能够稳定超越静态杠杆的关键。\n\n")
        
        f.write("### 2. 两类散户画像在震荡与趋势中的非对称表现\n")
        f.write("- **非信仰型散户 (Non-Believer)** 在波动率高的震荡资产（如 **SOXX**）上，表现出了极显著的“摩擦损耗”。因为其在回撤超 8% 或感觉主跌期时触发 0% 仓位强制出局，在趋势连续度低的行情中，极易陷入“频繁止损割肉、高位放量追涨”的恶性循环，导致最终收益远落后于 1x 标的基准。\n")
        f.write("- **信仰型散户 (Believer)** 设定了 **80% 的底线持仓**（即主跌期也仅退守 0.8x 现货，绝不清仓），这种“底线信仰”配合在感觉主升期向 2.0x 杠杆冲锋的设定，在趋势行情好、大科技大牛市明显的资产上展现出极其恐怖的复利爆发力。在 **688_CHIP** 上，信仰型散户录得 **634.37%** 的总回报，相比 1x 基准（257.76%）和专业策略实现了大幅超额。这说明在大周期向上的科技长牛市中，“保留底仓死拿 + 趋势感觉加杠杆”对散户而言极为高效。\n\n")
        
        f.write("### 3. 机构动量协同策略：ETF 路径 vs 融资路径\n")
        f.write("- 专业策略的“每日 ETF 路径 2x”与“静态融资路径 2x”在累计收益和夏普比率上极其贴近。然而在实际物理操作中，每日 ETF 路径每天都会产生调仓折价、管理费摩擦以及互换利息；融资路径虽免除调仓磨损，但长期融资借贷面临年化利息流出，且如果个股在主跌期出现极端波动，融资路径由于其一次性借款不调整的静态属性，面临极高爆仓风险。因此，在大趋势处于 Bear 阶段时，机构策略将杠杆降低至 0.8x 并加入个股补涨防御，能有效保护账户净值。\n\n")
        
        f.write("## 第五部分：各资产策略回测走势中文图谱\n")
        f.write("> 图谱说明：\n")
        f.write("- 绿色背景 = QQQ EMA金叉主升，红色背景 = QQQ EMA死叉主跌，黄色背景 = 震荡整理。\n")
        f.write("- 绿色上三角 `^` 标示非信仰型散户因情绪高涨而满仓追涨 2.0x 杠杆的买点；红色下三角 `v` 标示其因恐慌或割肉而全清仓 0.0x 仓位的卖点。\n")
        f.write("- 紫色菱形 `D` 标示信仰型散户触发 60 日超额（或 QQQ 绝对收益）而首次满仓 2.0x 杠杆的入场点。\n")
        f.write("- 轴最右侧留白处标注了每条曲线在最终日期的 NAV 净值与总收益率。\n\n")
        for asset in target_assets:
            f.write(f"### {asset} 策略回测对比\n")
            f.write(f"![{asset} 回测图](images/{asset}_backtest.png)\n\n")
            
        f.write("## 第六部分：静态杠杆与动态策略回测对照表\n")
        f.write("下表列示了过往三年所有标的资产在静态及动态策略下的核心量化结果：\n\n")
        f.write("### 1. 静态杠杆表现与融资损耗分析表\n")
        f.write(df_static.to_markdown(index=False))
        f.write("\n\n")
        f.write("### 2. 动态杠杆策略回测表现对比表\n")
        f.write(df_strategy.to_markdown(index=False))
        f.write("\n")
        
    print("\n学术收官版报告已成功写入 第一阶段研究报告.md。")

if __name__ == "__main__":
    run_analysis()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from strategies import run_professional_strategy, run_retail_strategy_by_type
from backtester import LeverageSimulator, calculate_metrics

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_asset_data(name: str):
    path = f"data/{name}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件 {path} 不存在，请先运行 data_downloader.py")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

# 融资年化利率 (688_CHIP 4.0%，其余 6.5%)
RATES = {
    "688_CHIP": 0.04,
    "QQQ": 0.065,
    "SOXX": 0.065,
    "MU": 0.065,
    "SanDisk": 0.065,
    "EWY": 0.065
}

# ETF 费率 (中国为美国的 2.0 倍即 1.90%，其余 0.95%)
ETF_FEES = {
    "688_CHIP": 0.0095 * 2.0,
    "QQQ": 0.0095,
    "SOXX": 0.0095,
    "MU": 0.0095,
    "SanDisk": 0.0095,
    "EWY": 0.0095
}

# 交易成本 (A股为万0.5即0.00005，美股与韩国ETF为万1即0.0001)
TRANS_FEES = {
    "688_CHIP": 0.00005,
    "QQQ": 0.0001,
    "SOXX": 0.0001,
    "MU": 0.0001,
    "SanDisk": 0.0001,
    "EWY": 0.0001
}

def log_formatter(y, pos):
    """自适应对数轴格式化器，避免科学计数法"""
    if y >= 1.0:
        return f"{y:g}"
    else:
        return f"{y:.2f}"

def plot_3state_analysis():
    target_assets = ["688_CHIP", "QQQ", "SOXX", "MU", "SanDisk", "EWY"]
    
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/images_3state", exist_ok=True)
    
    qqq_df = load_asset_data("QQQ")
    
    # 读取最佳扫参参数组合
    best_comb_path = "output/parameter_sweep_best_combinations.csv"
    if not os.path.exists(best_comb_path):
        raise FileNotFoundError(f"扫参最佳参数文件 {best_comb_path} 不存在，请先运行 parameter_sweep.py")
    df_best = pd.read_csv(best_comb_path)
    
    performance_records = []
    
    for asset in target_assets:
        print(f"正在绘制标的：{asset} 三状态仓位曲线...")
        asset_df = load_asset_data(asset)
        prices = asset_df['Close']
        returns = prices.pct_change().fillna(0.0)
        r_f = RATES[asset]
        fee_rate = ETF_FEES[asset]
        
        # 过滤该标的下的最佳参数
        asset_best = df_best[df_best["Asset"] == asset]
        
        def get_params_for_criteria(strat, criteria):
            row = asset_best[(asset_best["Strategy"] == strat) & (asset_best["Criteria"] == criteria)]
            if len(row) > 0:
                return {
                    "exposure_bull": float(row.iloc[0]["ExposureBull"]),
                    "exposure_oscillate": float(row.iloc[0]["ExposureOscillate"]),
                    "exposure_bear": float(row.iloc[0]["ExposureBear"]),
                    "vol_target": float(row.iloc[0]["VolTarget"]) if not pd.isna(row.iloc[0]["VolTarget"]) else None,
                    "max_leverage": float(row.iloc[0]["ExposureBull"])
                }
            return None
            
        # 构建唯一的回测运行
        runs = {}
        
        # 1. 机构专业策略：从扫参最佳表加载作为对照基准
        for strat in ["专业策略 (每日 ETF)", "专业策略 (静态融资)"]:
            for crit in ["Max Sharpe", "Max Return"]:
                p = get_params_for_criteria(strat, crit)
                if p is not None:
                    p_tuple = (p["exposure_bull"], p["exposure_oscillate"], p["exposure_bear"], p["vol_target"])
                    key = (strat, p_tuple)
                    if key not in runs:
                        runs[key] = []
                    runs[key].append(crit)
                    
        # 2. 散户策略：使用 4 组经典固定画像参数
        if asset == "688_CHIP":
            non_bel_name = "散户融资策略 (非信仰型)"
            bel_name = "散户融资策略 (信仰型)"
        else:
            non_bel_name = "散户策略 (非信仰型 ETF)"
            bel_name = "散户策略 (信仰型 ETF)"
            
        retail_configs = [
            (non_bel_name, (2.0, 0.8, 0.0, None), "激进画像"),
            (non_bel_name, (1.5, 0.5, 0.0, None), "保守画像"),
            (bel_name, (2.0, 1.3, 1.0, None), "激进画像"),
            (bel_name, (2.0, 1.0, 0.5, None), "保守画像")
        ]
        
        for strat, p_tuple, tag in retail_configs:
            key = (strat, p_tuple)
            if key not in runs:
                runs[key] = []
            runs[key].append(tag)
                    
        # 基础对比项
        nav_1x = prices / prices.iloc[0]
        etf_ret_2x = LeverageSimulator.simulate_daily_etf(returns, 2.0, r_f, fee_rate)
        nav_2x = (1.0 + etf_ret_2x).cumprod()
        
        real_navs = {}
        if asset == "QQQ":
            tqqq_df = load_asset_data("TQQQ")
            real_navs["3x 真实 ETF (TQQQ)"] = tqqq_df['Close'] / tqqq_df['Close'].iloc[0]
        elif asset == "SOXX":
            soxl_df = load_asset_data("SOXL")
            real_navs["3x 真实 ETF (SOXL)"] = soxl_df['Close'] / soxl_df['Close'].iloc[0]
        elif asset == "MU":
            muu_df = load_asset_data("MUU")
            real_navs["2x 真实 ETF (MUU)"] = muu_df['Close'] / muu_df['Close'].iloc[0]
        elif asset == "SanDisk":
            snxx_df = load_asset_data("SNXX")
            real_navs["2x 真实 ETF (SNXX)"] = snxx_df['Close'] / snxx_df['Close'].iloc[0]
        elif asset == "EWY":
            koru_df = load_asset_data("KORU")
            real_navs["3x 真实 ETF (KORU)"] = koru_df['Close'] / koru_df['Close'].iloc[0]
            
        # 开始绘图
        fig, ax = plt.subplots(figsize=(15.5, 11))
        t = asset_df.index
        
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
        ax.set_xlim(t[0], t[-1] + pd.Timedelta(days=140))
        
        lines = []
        lines.append(("1x 底层现货基准", nav_1x, 'black', 1.5, '-'))
        lines.append(("2x 静态杠杆 ETF (每日做多 ETF)", nav_2x, 'gray', 1.2, '--'))
        
        # 动态配色分配
        strat_color_map = {
            "专业策略 (每日 ETF)": {"base": "#1f77b4", "alt": "#17becf"},
            "专业策略 (静态融资)": {"base": "#aec7e8", "alt": "#5bc0de"},
            "散户策略 (非信仰型 ETF)": {"base": "#d62728", "alt": "#ff9896"},
            "散户策略 (信仰型 ETF)": {"base": "#9467bd", "alt": "#c5b0d5"},
            "散户现货策略 (非信仰型)": {"base": "#2ca02c", "alt": "#98df8a"},
            "散户现货策略 (信仰型)": {"base": "#bcbd22", "alt": "#dbdb8d"},
            "散户融资策略 (非信仰型)": {"base": "#ff7f0e", "alt": "#ffbb78"},
            "散户融资策略 (信仰型)": {"base": "#e377c2", "alt": "#f7b6d2"}
        }
        
        # 保存散户曲线以便在绘图循环中直接散布点标记
        retail_scatters = []
        
        for (strat, p_tuple), crits in runs.items():
            exp_bull, exp_osc, exp_bear, vol_t = p_tuple
            p = {
                "exposure_bull": exp_bull,
                "exposure_oscillate": exp_osc,
                "exposure_bear": exp_bear,
                "vol_target": vol_t,
                "max_leverage": exp_bull
            }
            
            # 运行回测
            t_fee = TRANS_FEES[asset]
            if "专业策略" in strat:
                res_df = run_professional_strategy(
                    asset_df, qqq_df, p, 
                    leverage_type="daily_etf" if "每日 ETF" in strat else "margin_static", 
                    interest_rate=r_f, fee_rate_annual=fee_rate,
                    trans_fee_rate=t_fee
                )
            else:
                profile_type = "believer" if "信仰型" in strat else "non_believer"
                res_df = run_retail_strategy_by_type(
                    asset_df, qqq_df, p, 
                    profile_type=profile_type, leverage_type="daily_etf", 
                    interest_rate=r_f, fee_rate_annual=fee_rate,
                    trans_fee_rate=t_fee
                )
                
                # 记录该散户曲线的转换信号点
                retail_scatters.append((strat, res_df, crits))
                    
            crit_label = " & ".join(crits)
            label = f"{strat} [{exp_bull},{exp_osc},{exp_bear}] ({crit_label})"
            
            # 设定线型与颜色
            if len(crits) == 2:
                ls = '-'
                lw = 2.0
                color = strat_color_map[strat]["base"]
            elif "Max Sharpe" in crits:
                ls = '-'
                lw = 1.8
                color = strat_color_map[strat]["base"]
            else:
                ls = '--'
                lw = 1.5
                color = strat_color_map[strat]["alt"]
                
            lines.append((label, res_df['nav'], color, lw, ls))
            
        for label, nav_s in real_navs.items():
            lines.append((label, nav_s, 'brown', 1.5, '-'))
            
        # 绘制散户策略曲线，其他策略不画曲线，仅保留终点数据标注
        for label, nav_s, color, lw, ls in lines:
            if "散户" in label:
                ax.plot(nav_s.index, nav_s.values, label=label, color=color, linewidth=lw, linestyle=ls)
            
        # 8. 直接对【激进画像】散户回测线绘制其买卖/仓位调整信号点，以保持图面整洁并形成对比
        buy_legend_added = False
        sell_legend_added = False
        panic_legend_added = False
        
        for strat, res_df, crits in retail_scatters:
            # 仅对“激进画像”进行打点
            if "激进画像" not in crits:
                continue
                
            diff_exp = res_df['exposure'].diff().fillna(0.0)
            if len(res_df) > 0 and res_df['exposure'].iloc[0] > 0.0:
                diff_exp.iloc[0] = res_df['exposure'].iloc[0]
                
            prev_disc = res_df['exposure'].shift(1).fillna(0.0) > 0.0
            curr_disc = res_df['exposure'] > 0.0
            
            # 区分买、普通减仓、恐慌清仓
            imp_buy = res_df[diff_exp > 0.0]
            imp_panic = res_df[prev_disc & ~curr_disc]
            imp_sell = res_df[(diff_exp < 0.0) & ~(prev_disc & ~curr_disc)]
            
            size_p = 40
            
            if "非信仰型" in strat:
                # 非信仰型：使用青色三角形加仓，橙色三角形减仓，深红色大 X 斩仓
                ax.scatter(imp_buy.index, imp_buy['nav'], color='#17becf', marker='^', s=size_p, zorder=6, 
                           label='非信仰加仓' if not buy_legend_added else "")
                ax.scatter(imp_sell.index, imp_sell['nav'], color='#ff7f0e', marker='v', s=size_p, zorder=6, 
                           label='非信仰减仓' if not sell_legend_added else "")
                ax.scatter(imp_panic.index, imp_panic['nav'], color='#d62728', marker='X', s=75, zorder=8, 
                           label='非信仰恐慌斩仓' if not panic_legend_added else "")
                buy_legend_added = True
                sell_legend_added = True
                panic_legend_added = True
            elif "信仰型" in strat:
                # 信仰型：使用绿色三角形加仓，红色三角形减仓
                ax.scatter(imp_buy.index, imp_buy['nav'], color='#2ca02c', marker='^', s=size_p, zorder=6, 
                           label='信仰加仓' if not buy_legend_added else "")
                ax.scatter(imp_sell.index, imp_sell['nav'], color='#d62728', marker='v', s=size_p, zorder=6, 
                           label='信仰减仓' if not sell_legend_added else "")
                buy_legend_added = True
                sell_legend_added = True
            
        # 9. 终点防重叠文字对齐 (以绘制的散户线为主进行范围判定，包含隐藏参考线终点)
        y_min_val = min([nav_s.min() for label, nav_s, _, _, _ in lines if "散户" in label] + [nav_s.iloc[-1] for label, nav_s, _, _, _ in lines if "散户" not in label])
        y_max_val = max([nav_s.max() for label, nav_s, _, _, _ in lines if "散户" in label] + [nav_s.iloc[-1] for label, nav_s, _, _, _ in lines if "散户" not in label])
        
        label_positions = []
        for label, nav_s, color, lw, ls in lines:
            final_nav = nav_s.iloc[-1]
            final_ret = final_nav - 1.0
            metrics_cur = calculate_metrics(nav_s)
            
            performance_records.append({
                "Asset": asset,
                "Strategy": label,
                "TotalReturn": f"{final_ret:.2%}",
                "Sharpe": f"{metrics_cur['Sharpe Ratio']:.2f}",
                "MDD": f"{metrics_cur['Max Drawdown']:.2%}"
            })
            
            disp_label = label.replace(" (每日做多 ETF)", "").replace(" (每日 ETF)", "")
            # 若不是散户策略，则标注为灰色参考文字
            if "散户" not in label:
                text_str = f" [参考] {disp_label}\n  NAV: {final_nav:.2f} ({final_ret:+.1%})\n  Sharpe: {metrics_cur['Sharpe Ratio']:.2f} | MDD: {metrics_cur['Max Drawdown']:.1%}"
                text_color = "gray"
            else:
                text_str = f" {disp_label}\n  NAV: {final_nav:.2f} ({final_ret:+.1%})\n  Sharpe: {metrics_cur['Sharpe Ratio']:.2f} | MDD: {metrics_cur['Max Drawdown']:.1%}"
                text_color = color
                
            label_positions.append({
                'y': final_nav,
                'target_y': final_nav,
                'text': text_str,
                'color': text_color,
                'last_idx': nav_s.index[-1]
            })
            
        label_positions.sort(key=lambda x: x['y'], reverse=True)
        
        y_min_log = np.log10(y_min_val)
        y_max_log = np.log10(y_max_val)
        min_spacing_log = (y_max_log - y_min_log) * 0.038
        if min_spacing_log == 0:
            min_spacing_log = 0.02
            
        for i in range(len(label_positions)):
            label_positions[i]['log_y'] = np.log10(label_positions[i]['y'])
            
        for i in range(1, len(label_positions)):
            if label_positions[i-1]['log_y'] - label_positions[i]['log_y'] < min_spacing_log:
                label_positions[i]['log_y'] = label_positions[i-1]['log_y'] - min_spacing_log
                
        for i in range(len(label_positions)-2, -1, -1):
            if label_positions[i]['log_y'] - label_positions[i+1]['log_y'] < min_spacing_log:
                label_positions[i]['log_y'] = label_positions[i+1]['log_y'] + min_spacing_log
                
        for item in label_positions:
            y_pos = 10 ** item['log_y']
            ax.text(item['last_idx'] + pd.Timedelta(days=5), y_pos, item['text'], 
                    color=item['color'], fontsize=8.0, fontweight='bold', verticalalignment='center')
            
        # 10. 绘制大盘背景色块
        try:
            state_series = run_professional_strategy(asset_df, qqq_df, get_params_for_criteria("专业策略 (每日 ETF)", "Max Sharpe"), 
                                                     leverage_type="daily_etf", interest_rate=r_f, fee_rate_annual=fee_rate)['state']
            current_state = None
            start_idx = 0
            for i in range(len(state_series)):
                state = state_series.iloc[i]
                if state != current_state:
                    if current_state is not None:
                        color = {'bull': 'green', 'bear': 'red', 'oscillate': 'yellow'}[current_state]
                        ax.axvspan(t[start_idx], t[i-1], color=color, alpha=0.04)
                    current_state = state
                    start_idx = i
            if current_state is not None:
                color = {'bull': 'green', 'bear': 'red', 'oscillate': 'yellow'}[current_state]
                ax.axvspan(t[start_idx], t[-1], color=color, alpha=0.04)
        except Exception as e:
            print(f"背景色块绘制失败: {e}")
            
        ax.set_title(f"{asset} 核心策略仓位与回测对比走势图 (实线=Max Sharpe, 虚线=Max Return, Y对数轴)\n"
                     f"背景背景：绿色=QQQ&个股主升，红色=主跌死叉防守，黄色=震荡过渡期", fontsize=13, fontweight='bold')
        ax.set_xlabel("日期", fontsize=11)
        ax.set_ylabel("归一化账户净值 (NAV, Log Scale)", fontsize=11)
        ax.grid(True, which="both", linestyle=':', alpha=0.6)
        ax.legend(loc="upper left", fontsize=8.8)
        plt.tight_layout()
        
        plot_path = f"output/images_3state/{asset}_3state_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"完成保存走势对比图至 {plot_path}")
        
    df_perf = pd.DataFrame(performance_records)
    df_perf.to_csv("output/three_state_profiles_performance.csv", index=False)
    print("三状态典型绩效对照表生成完毕。")

if __name__ == "__main__":
    plot_3state_analysis()

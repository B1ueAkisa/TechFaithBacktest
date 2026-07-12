import os
import itertools
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from strategies import run_professional_strategy, run_retail_strategy_by_type
from backtester import calculate_metrics

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

def evaluate_single_run(asset, strat_name, exp_bull, exp_osc, exp_bear, vol_target, lev_type):
    try:
        asset_df = load_asset_data(asset)
        qqq_df = load_asset_data("QQQ")
        r_f = RATES[asset]
        fee_rate = ETF_FEES[asset]
        
        # 构建统一的 3 状态仓位参数
        params = {
            'exposure_bull': exp_bull,
            'exposure_oscillate': exp_osc,
            'exposure_bear': exp_bear,
            'max_leverage': exp_bull  # 兼容旧 max_leverage 语义作上限
        }
        
        if "专业策略" in strat_name:
            params['ema_fast'] = 20
            params['ema_slow'] = 50
            params['vol_target'] = vol_target
            res_df = run_professional_strategy(
                asset_df, qqq_df, params, 
                leverage_type=lev_type, 
                interest_rate=r_f, 
                fee_rate_annual=fee_rate
            )
        else:
            profile_type = 'non_believer' if "非信仰型" in strat_name else 'believer'
            res_df = run_retail_strategy_by_type(
                asset_df, qqq_df, params, 
                profile_type=profile_type, 
                leverage_type=lev_type, 
                interest_rate=r_f, 
                fee_rate_annual=fee_rate
            )
            
        metrics = calculate_metrics(res_df['nav'])
        return {
            "Asset": asset,
            "Strategy": strat_name,
            "LeverageType": lev_type,
            "Criteria": "", # 后续填充
            "ExposureBull": exp_bull,
            "ExposureOscillate": exp_osc,
            "ExposureBear": exp_bear,
            "VolTarget": vol_target if vol_target is not None else np.nan,
            "TotalReturn": metrics["Total Return"],
            "AnnualReturn": metrics["Annual Return"],
            "Vol": metrics["Annual Volatility"],
            "Sharpe": metrics["Sharpe Ratio"],
            "MDD": metrics["Max Drawdown"],
            "Calmar": metrics["Calmar Ratio"] if not pd.isna(metrics["Calmar Ratio"]) else np.nan
        }
    except Exception as e:
        print(f"Error sweeping {asset} {strat_name} bull={exp_bull} osc={exp_osc} bear={exp_bear}: {e}")
        return None

def run_sweep():
    target_assets = ["688_CHIP", "QQQ", "SOXX", "MU", "SanDisk", "EWY"]
    vol_targets = [0.15, 0.25, 0.35, 0.45]
    
    # 扫参三状态杠杆网格空间定义
    grids_global = {
        "bull": [1.0, 1.5, 2.0, 2.5, 3.0],
        "oscillate": [0.5, 1.0, 1.5, 2.0, 2.5],
        "bear": [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]
    }
    
    grids_china = {
        "bull": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        "oscillate": [0.3, 0.5, 0.8, 1.0, 1.2, 1.5],
        "bear": [0.0, 0.3, 0.5, 0.8, 1.0]
    }
    
    tasks = []
    
    for asset in target_assets:
        grid = grids_china if asset == "688_CHIP" else grids_global
        
        # 1. 生成大前提：满足单调递减规则的三元组 (E_bull >= E_oscillate >= E_bear)
        triplets = []
        for b, o, r in itertools.product(grid["bull"], grid["oscillate"], grid["bear"]):
            if b >= o and o >= r:
                triplets.append((b, o, r))
                
        # 2. 专业策略（不设心理边界，仅满足递减约束）
        for b, o, r in triplets:
            for vol_t in vol_targets:
                tasks.append((asset, "专业策略 (每日 ETF)", b, o, r, vol_t, "daily_etf"))
                tasks.append((asset, "专业策略 (静态融资)", b, o, r, vol_t, "margin_static"))
                
        # 3. 散户策略（引入符合实际心理学的边界条件）
        if asset == "688_CHIP":
            # 中国市场细分为：现货散户 (b <= 1.0) 与 融资散户 (b > 1.0)
            for b, o, r in triplets:
                if b <= 1.0: # 现货散户
                    # 非信仰型散户现货：Bear 阶段必须完全割肉清仓 (r == 0.0) 以确保买卖点逻辑与一阶段完全一致
                    if r == 0.0:
                        tasks.append((asset, "散户现货策略 (非信仰型)", b, o, r, None, "daily_etf"))
                    # 信仰型散户现货：Bear 阶段必须死扛底仓，持仓不能低于 0.5 (r >= 0.5)
                    if r >= 0.5:
                        tasks.append((asset, "散户现货策略 (信仰型)", b, o, r, None, "daily_etf"))
                else: # 融资散户
                    # 非信仰型散户融资：Bear 阶段必须完全割肉清仓 (r == 0.0)
                    if r == 0.0:
                        tasks.append((asset, "散户融资策略 (非信仰型)", b, o, r, None, "daily_etf"))
                    # 信仰型散户融资：Bear 阶段必须死扛底仓，融资仓位不低于 0.5 (r >= 0.5)
                    if r >= 0.5:
                        tasks.append((asset, "散户融资策略 (信仰型)", b, o, r, None, "daily_etf"))
        else:
            # 全球市场散户
            for b, o, r in triplets:
                # 非信仰型散户全球：Bear 阶段必须完全割肉清仓 (r == 0.0)
                if r == 0.0:
                    tasks.append((asset, "散户策略 (非信仰型 ETF)", b, o, r, None, "daily_etf"))
                # 信仰型散户全球：Bear 阶段必须死扛底仓，仓位绝不能低于 0.8 (r >= 0.8)
                if r >= 0.8:
                    tasks.append((asset, "散户策略 (信仰型 ETF)", b, o, r, None, "daily_etf"))

    print(f"应用行为学约束后的三状态扫参总点数: {len(tasks)}。启动并行计算...")
    
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(evaluate_single_run, *t) for t in tasks]
        
        total = len(futures)
        for idx, fut in enumerate(futures):
            res = fut.result()
            if res is not None:
                results.append(res)
            if (idx + 1) % 1000 == 0:
                print(f"进度: {idx + 1}/{total} ({(idx + 1)/total:.1%}) 已完成")
                
    df_results = pd.DataFrame(results)
    os.makedirs("output", exist_ok=True)
    df_results.to_csv("output/parameter_sweep_results.csv", index=False)
    print("已完整保存扫参结果至 output/parameter_sweep_results.csv")
    
    # 提取最优夏普、卡玛与绝对收益
    print("\n------------------ 最优组合提取中 ------------------")
    best_records = []
    
    for asset in target_assets:
        asset_df = df_results[df_results["Asset"] == asset]
        for strat in asset_df["Strategy"].unique():
            strat_df = asset_df[asset_df["Strategy"] == strat]
            
            # Max Sharpe
            best_sharpe_row = strat_df.sort_values(by="Sharpe", ascending=False).iloc[0]
            best_records.append({
                "Asset": asset,
                "Strategy": strat,
                "LeverageType": best_sharpe_row["LeverageType"],
                "Criteria": "Max Sharpe",
                "ExposureBull": best_sharpe_row["ExposureBull"],
                "ExposureOscillate": best_sharpe_row["ExposureOscillate"],
                "ExposureBear": best_sharpe_row["ExposureBear"],
                "VolTarget": best_sharpe_row["VolTarget"],
                "TotalReturn": f"{best_sharpe_row['TotalReturn']:.2%}",
                "AnnualReturn": f"{best_sharpe_row['AnnualReturn']:.2%}",
                "Vol": f"{best_sharpe_row['Vol']:.2%}",
                "Sharpe": f"{best_sharpe_row['Sharpe']:.2f}",
                "MDD": f"{best_sharpe_row['MDD']:.2%}",
                "Calmar": f"{best_sharpe_row['Calmar']:.2f}" if not pd.isna(best_sharpe_row['Calmar']) else "N/A"
            })
            
            # Max Calmar
            best_calmar_df = strat_df.dropna(subset=["Calmar"])
            if len(best_calmar_df) > 0:
                best_calmar_row = best_calmar_df.sort_values(by="Calmar", ascending=False).iloc[0]
                best_records.append({
                    "Asset": asset,
                    "Strategy": strat,
                    "LeverageType": best_calmar_row["LeverageType"],
                    "Criteria": "Max Calmar",
                    "ExposureBull": best_calmar_row["ExposureBull"],
                    "ExposureOscillate": best_calmar_row["ExposureOscillate"],
                    "ExposureBear": best_calmar_row["ExposureBear"],
                    "VolTarget": best_calmar_row["VolTarget"],
                    "TotalReturn": f"{best_calmar_row['TotalReturn']:.2%}",
                    "AnnualReturn": f"{best_calmar_row['AnnualReturn']:.2%}",
                    "Vol": f"{best_calmar_row['Vol']:.2%}",
                    "Sharpe": f"{best_calmar_row['Sharpe']:.2f}",
                    "MDD": f"{best_calmar_row['MDD']:.2%}",
                    "Calmar": f"{best_calmar_row['Calmar']:.2f}"
                })
                
            # Max Return
            best_return_row = strat_df.sort_values(by="TotalReturn", ascending=False).iloc[0]
            best_records.append({
                "Asset": asset,
                "Strategy": strat,
                "LeverageType": best_return_row["LeverageType"],
                "Criteria": "Max Return",
                "ExposureBull": best_return_row["ExposureBull"],
                "ExposureOscillate": best_return_row["ExposureOscillate"],
                "ExposureBear": best_return_row["ExposureBear"],
                "VolTarget": best_return_row["VolTarget"],
                "TotalReturn": f"{best_return_row['TotalReturn']:.2%}",
                "AnnualReturn": f"{best_return_row['AnnualReturn']:.2%}",
                "Vol": f"{best_return_row['Vol']:.2%}",
                "Sharpe": f"{best_return_row['Sharpe']:.2f}",
                "MDD": f"{best_return_row['MDD']:.2%}",
                "Calmar": f"{best_return_row['Calmar']:.2f}" if not pd.isna(best_return_row['Calmar']) else "N/A"
            })
            
    df_best = pd.DataFrame(best_records)
    df_best.to_csv("output/parameter_sweep_best_combinations.csv", index=False)
    print("最优三状态参数组合已写入 output/parameter_sweep_best_combinations.csv")

if __name__ == "__main__":
    run_sweep()

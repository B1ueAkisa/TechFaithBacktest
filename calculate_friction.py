import sys
import os
sys.path.append("d:/AKFile/其他/投资/杠杆与收益")
import pandas as pd
import numpy as np
from strategies import run_retail_strategy_by_type

target_assets = ["688_CHIP", "QQQ", "SOXX", "MU", "SanDisk", "EWY"]

RATES = {
    "688_CHIP": 0.04,
    "QQQ": 0.065,
    "SOXX": 0.065,
    "MU": 0.065,
    "SanDisk": 0.065,
    "EWY": 0.065
}

ETF_FEES = {
    "688_CHIP": 0.019,
    "QQQ": 0.0095,
    "SOXX": 0.0095,
    "MU": 0.0095,
    "SanDisk": 0.0095,
    "EWY": 0.0095
}

TRANS_FEES = {
    "688_CHIP": 0.00005,
    "QQQ": 0.0001,
    "SOXX": 0.0001,
    "MU": 0.0001,
    "SanDisk": 0.0001,
    "EWY": 0.0001
}

def count_exposure_changes(exp_series):
    changes = (exp_series.diff().fillna(0.0) != 0.0).sum()
    if len(exp_series) > 0 and exp_series.iloc[0] != 0.0:
        changes += 1
    return int(changes)

def run_friction_analysis(comparison_type):
    # 区分激进与保守参数组合
    if comparison_type == "aggressive":
        p_non = {"exposure_bull": 2.0, "exposure_oscillate": 0.8, "exposure_bear": 0.0}
        p_bel = {"exposure_bull": 2.0, "exposure_oscillate": 1.3, "exposure_bear": 1.0}
    else:
        p_non = {"exposure_bull": 1.5, "exposure_oscillate": 0.5, "exposure_bear": 0.0}
        p_bel = {"exposure_bull": 2.0, "exposure_oscillate": 1.0, "exposure_bear": 0.5}
        
    records = []
    
    for asset in target_assets:
        asset_df = pd.read_csv(f"data/{asset}.csv", index_col=0, parse_dates=True)
        qqq_df = pd.read_csv(f"data/QQQ.csv", index_col=0, parse_dates=True)
        
        r_f = RATES[asset]
        fee_rate = ETF_FEES[asset]
        t_fee = TRANS_FEES[asset]
        
        res_non = run_retail_strategy_by_type(
            asset_df, qqq_df, p_non, 
            profile_type="non_believer", leverage_type="daily_etf",
            interest_rate=r_f, fee_rate_annual=fee_rate, trans_fee_rate=t_fee
        )
        res_bel = run_retail_strategy_by_type(
            asset_df, qqq_df, p_bel, 
            profile_type="believer", leverage_type="daily_etf",
            interest_rate=r_f, fee_rate_annual=fee_rate, trans_fee_rate=t_fee
        )
        
        trades_non = count_exposure_changes(res_non['exposure'])
        trades_bel = count_exposure_changes(res_bel['exposure'])
        
        cash_ratio_non = (res_non['exposure'] == 0.0).sum() / len(res_non)
        cash_ratio_bel = (res_bel['exposure'] == 0.0).sum() / len(res_bel)
        
        asset_returns = asset_df['Close'].pct_change().fillna(0.0)
        cash_periods_non = res_non['exposure'] == 0.0
        missed_return = (1.0 + asset_returns[cash_periods_non]).prod() - 1.0
        
        records.append({
            "Asset": asset,
            "Non_Trades": trades_non,
            "Bel_Trades": trades_bel,
            "Non_Cash_Ratio": f"{cash_ratio_non:.1%}",
            "Bel_Cash_Ratio": f"{cash_ratio_bel:.1%}",
            "Non_Missed_Return": f"{missed_return:+.2%}",
            "Non_NAV": res_non['nav'].iloc[-1],
            "Bel_NAV": res_bel['nav'].iloc[-1],
            "Premium": f"{(res_bel['nav'].iloc[-1] - res_non['nav'].iloc[-1]):+.2f}"
        })
        
    df = pd.DataFrame(records)
    output_path = f"output/retail_friction_analysis_{comparison_type}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n--- 摩擦数据对照表 ({comparison_type}) ---")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    run_friction_analysis("aggressive")
    run_friction_analysis("conservative")

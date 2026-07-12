import numpy as np
import pandas as pd
from backtester import LeverageSimulator, MarginAccount

def get_qqq_global_state(qqq_df: pd.DataFrame, target_index: pd.Index, ema_fast_span: int = 20, ema_slow_span: int = 50):
    """
    基于 QQQ 的 EMA 均线过滤宏观趋势（防止未来函数，shift(1)）。
    """
    # 1. 在完整的 QQQ 数据上计算 EMA，避免因标的时间偏短（如SNDK）导致边界失真
    ema_fast_full = qqq_df['Close'].ewm(span=ema_fast_span, adjust=False).mean()
    ema_slow_full = qqq_df['Close'].ewm(span=ema_slow_span, adjust=False).mean()
    
    is_bull_full = (qqq_df['Close'] > ema_fast_full) & (ema_fast_full > ema_slow_full)
    is_bear_full = (qqq_df['Close'] < ema_fast_full) & (ema_fast_full < ema_slow_full)
    is_oscillate_full = ~(is_bull_full | is_bear_full)
    
    # 2. 将计算好的信号对齐到 target_index
    is_bull = is_bull_full.reindex(target_index).ffill().fillna(False)
    is_bear = is_bear_full.reindex(target_index).ffill().fillna(False)
    is_oscillate = is_oscillate_full.reindex(target_index).ffill().fillna(True)
    qqq_close = qqq_df['Close'].reindex(target_index).ffill()
    
    signals = pd.DataFrame({
        'is_bull': is_bull.shift(1).fillna(False),
        'is_bear': is_bear.shift(1).fillna(False),
        'is_oscillate': is_oscillate.shift(1).fillna(True),
        'qqq_close': qqq_close
    }, index=target_index)
    
    return signals


def run_professional_strategy(asset_df: pd.DataFrame, qqq_df: pd.DataFrame, params: dict, 
                              leverage_type: str = 'daily_etf', interest_rate: float = 0.04,
                              fee_rate_annual: float = 0.0095) -> pd.DataFrame:
    """
    机构级专业策略 (大盘与个股协同杠杆控制模型)
    - 结合大盘 QQQ 状态与个股走势 (个股收盘价是否大于自身 20 EMA) 划分三大阶段：
      - 主升期 (Bull): QQQ主升 且 个股 > 20EMA.
      - 主跌期 (Bear): QQQ主跌 且 个股 < 20EMA.
      - 震荡期 (Oscillate): 其余情况.
    - 引入 60 日个股与大盘相对超额收益 ($Rel_{60d} = R^{\text{asset}}_{60d} - R^{\text{QQQ}}_{60d}$):
      - 若 $Rel_{60d} < -10\%$ 视为“落后补涨”状态：若处于主跌期，提高仓位至 1.0x (防守性增强)。
      - 若 $Rel_{60d} > 10\%$ 视为“超涨”状态：若处于主升或震荡期，杠杆低配（仓位下调 0.3x）。
    - 波动率风控因子 $F_{\text{vol}}$ 动态乘数继续生效。
    """
    target_index = asset_df.index
    prices = asset_df['Close']
    returns = prices.pct_change().fillna(0.0)
    
    # 大盘趋势
    signals = get_qqq_global_state(
        qqq_df, target_index, 
        ema_fast_span=params.get('ema_fast', 20), 
        ema_slow_span=params.get('ema_slow', 50)
    )
    
    # 个股均线趋势 (shift(1))
    asset_ema20 = prices.ewm(span=20, adjust=False).mean().shift(1).fillna(prices.iloc[0])
    asset_above_ema20 = prices.shift(1).fillna(prices.iloc[0]) > asset_ema20
    
    # 60日相对超额 (shift(1))
    asset_ret_60d = prices.pct_change(60).shift(1).fillna(0.0)
    qqq_close = qqq_df['Close'].reindex(target_index).ffill()
    qqq_ret_60d = qqq_close.pct_change(60).shift(1).fillna(0.0)
    rel_return_60d = asset_ret_60d - qqq_ret_60d
    
    # 20日滚动波动率，shift(1)
    rolling_vol = (returns.rolling(window=20).std() * np.sqrt(252)).shift(1).fillna(params.get('vol_target', 0.25))
    
    max_lev = params.get('max_leverage', 2.0)
    vol_target = params.get('vol_target', 0.25)
    r_cash = 0.02 / 252
    
    exposures = []
    states = []
    state_nums = []
    
    for idx, row in signals.iterrows():
        vol = rolling_vol.loc[idx]
        f_vol = max(0.0, 1.0 - (vol / vol_target))
        
        # 个股均线状态
        above_ema = asset_above_ema20.loc[idx]
        
        # 行情识别：大盘与个股协同
        exp_bear = params.get('exposure_bear', 0.8)
        exp_bull = params.get('exposure_bull', max_lev)
        exp_osc = params.get('exposure_oscillate', 1.0 + (params.get('oscillate_weight', 0.7) * (max_lev - 1.0)))
        
        if row['is_bear'] and (not above_ema):
            # 主跌期
            base_exp = exp_bear
            state = 'bear'
            num = 0
        elif row['is_bull'] and above_ema:
            # 主升期
            base_exp = 1.0 + (exp_bull - 1.0) * f_vol if exp_bull > 1.0 else exp_bull * f_vol
            state = 'bull'
            num = 2
        else:
            # 震荡期
            base_exp = 1.0 + (exp_osc - 1.0) * f_vol if exp_osc > 1.0 else exp_osc * f_vol
            state = 'oscillate'
            num = 1
            
        # 相对超额调节
        rel_ret = rel_return_60d.loc[idx]
        exp = base_exp
        
        if rel_ret < -0.10: # 落后补涨
            if state == 'bear':
                exp = 1.0 # 主跌期仓位高一点，保留 1.0x 现货
        elif rel_ret > 0.10: # 超涨
            if state in ['bull', 'oscillate']:
                exp = max(1.0, base_exp - 0.3) # 杠杆低配 (仓位下调0.3)
                
        exposures.append(exp)
        states.append(state)
        state_nums.append(num)
        
    exposures = pd.Series(exposures, index=target_index)
    state_nums = pd.Series(state_nums, index=target_index)
    
    state_change = state_nums.diff().fillna(0.0)
    transitions = state_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    portfolio_returns = []
    
    if leverage_type == 'daily_etf':
        etf_ret = LeverageSimulator.simulate_daily_etf(returns, max_lev, interest_rate, fee_rate_annual)
        for t in range(len(target_index)):
            exp = exposures.iloc[t]
            ret_t = returns.iloc[t]
            
            if exp <= 1.0:
                r_p = exp * ret_t + (1.0 - exp) * r_cash
            else:
                if max_lev > 1.0:
                    w_etf = (exp - 1.0) / (max_lev - 1.0)
                else:
                    w_etf = 0.0
                w_1x = 1.0 - w_etf
                r_p = w_1x * ret_t + w_etf * etf_ret.iloc[t]
            portfolio_returns.append(r_p)
            
    elif leverage_type == 'margin_constant':
        for t in range(len(target_index)):
            exp = exposures.iloc[t]
            ret_t = returns.iloc[t]
            if exp <= 1.0:
                r_p = exp * ret_t + (1.0 - exp) * r_cash
            else:
                interest_daily = interest_rate / 365.0
                friction = exp * abs(exp - 1.0) * abs(ret_t) * 0.001
                r_p = exp * ret_t - (exp - 1.0) * interest_daily - friction
            portfolio_returns.append(r_p)
            
    elif leverage_type == 'margin_static':
        account = MarginAccount(initial_equity=100.0, interest_rate_annual=interest_rate, trans_fee_rate=0.001)
        prev_exp = 1.0
        nav_series = [100.0]
        
        for t in range(1, len(target_index)):
            exp = exposures.iloc[t]
            ret_t = returns.iloc[t]
            price_t_1 = prices.iloc[t-1]
            
            if exp != prev_exp:
                account.rebalance(exp, price_t_1)
                prev_exp = exp
                
            state = account.update_daily(ret_t)
            nav_series.append(state['equity'])
            
        nav_s = pd.Series(nav_series, index=target_index)
        portfolio_returns = nav_s.pct_change().fillna(0.0)
        
    portfolio_returns = pd.Series(portfolio_returns, index=target_index)
    nav_s = (1.0 + portfolio_returns).cumprod()
    
    result_df = pd.DataFrame({
        'nav': nav_s,
        'exposure': exposures,
        'state': states,
        'transition': transitions,
        'close': prices
    }, index=target_index)
    
    return result_df


def run_retail_strategy_by_type(asset_df: pd.DataFrame, qqq_df: pd.DataFrame, params: dict, 
                                profile_type: str = 'non_believer', leverage_type: str = 'daily_etf', 
                                interest_rate: float = 0.04, fee_rate_annual: float = 0.0095) -> pd.DataFrame:
    """
    运行细分的散户画像策略（情绪感觉仓位调节版）：
    
    【散户情绪感觉模型】：
      - 散户在昨日价格创新高 (过去20天高点) 或触发“一阳改三观”时，感觉市场进入【主升期】(bull)。
      - 散户在昨日价格创新低 (过去20天低点) 或触发“一阴改三观”时，感觉市场进入【主跌期】(bear)。
      - 其余时间散户感觉市场处于【震荡期】(oscillating)。
      
    1. non_believer (非信仰型 / 原一惊一乍型)：
       - 仓位波动范围：0% 到 200% (Exposure: 0.0 至 2.0).
       - 初始空仓起步 (0% 仓位).
       - 进场：10日收益率 > 10% 或感觉进入主升期时激活，买入 2.0x 杠杆。
       - 持仓调仓：
         - 感觉主升：Exposure = 2.0 (满杠杆)
         - 感觉震荡：Exposure = 1.0 (纯现货无杠杆)
         - 感觉主跌 / 回撤 > 8%：立刻空仓清零 (Exposure = 0.0, discovered = False)。
       
    2. believer (信仰型 / 原信仰死拿型)：
       - 仓位最低不低于 80% (Exposure: 0.8 至 2.0)。
       - 初始空仓起步 (0% 仓位)，等待发现条件激活。
       - 进场：若标的本身是基准物 (QQQ)，则以其自身 60日绝对收益率超过 15% 触发买入；其他资产则以跑赢 QQQ 超 15% 触发买入。
       - 持仓调仓：
         - 感觉主升：Exposure = 2.0 (满杠杆)
         - 感觉震荡：Exposure = 1.5 (仓位折中)
         - 感觉主跌：Exposure = 0.8 (由于信仰不灭，最低绝不低于 80% 仓位，不割肉清仓，死扛到底)
    """
    target_index = asset_df.index
    prices = asset_df['Close']
    volumes = asset_df['Volume']
    returns = prices.pct_change().fillna(0.0)
    
    qqq_close = qqq_df['Close'].reindex(target_index).ffill()
    max_lev = params.get('max_leverage', 2.0)
    r_cash = 0.02 / 252
    
    ret_60d_asset = prices.pct_change(60)
    ret_60d_bench = qqq_close.pct_change(60)
    ret_10d_asset = prices.pct_change(10)
    
    # 自动识别资产本身是否为 benchmark
    is_benchmark_asset = (ret_60d_asset - ret_60d_bench).abs().max() < 1e-6
    
    high_20d = prices.rolling(20).max()
    low_20d = prices.rolling(20).min()
    vol_mean_20d = volumes.rolling(20).mean()
    
    discovered = False
    discovery_price_peak = 0.0
    
    exposures = []
    states = []
    transitions = []
    
    for t in range(len(target_index)):
        if t < 60:
            exposures.append(0.0)
            states.append('cash')
            transitions.append(0)
            continue
            
        # 基于前一日收盘数据 (t-1)
        price_t_1 = prices.iloc[t-1]
        volume_t_1 = volumes.iloc[t-1]
        ret_t_1 = returns.iloc[t-1]
        
        ret_60d_asset_t_1 = ret_60d_asset.iloc[t-1]
        ret_60d_bench_t_1 = ret_60d_bench.iloc[t-1]
        ret_10d_asset_t_1 = ret_10d_asset.iloc[t-1]
        
        high_20d_t_1 = high_20d.iloc[t-1]
        low_20d_t_1 = low_20d.iloc[t-1]
        vol_mean_t_1 = vol_mean_20d.iloc[t-1]
        
        is_extreme_up = (ret_t_1 > 0.04) and (volume_t_1 > 2.0 * vol_mean_t_1)
        is_extreme_down = (ret_t_1 < -0.04) and (volume_t_1 > 2.0 * vol_mean_t_1)
        
        feel_bull = (price_t_1 >= high_20d_t_1) or is_extreme_up
        feel_bear = (price_t_1 <= low_20d_t_1) or is_extreme_down
        
        if feel_bull and not feel_bear:
            feel_state = 'bull'
        elif feel_bear and not feel_bull:
            feel_state = 'bear'
        else:
            feel_state = 'oscillate'
            
        prev_discovered = discovered
        
        # 1. 状态跳转处理
        if profile_type == 'non_believer':
            if discovered:
                discovery_price_peak = max(discovery_price_peak, price_t_1)
                dd = (price_t_1 - discovery_price_peak) / discovery_price_peak
                if feel_state == 'bear' or (dd < -0.08):
                    discovered = False
            else:
                if feel_state == 'bull' or (ret_10d_asset_t_1 > 0.10):
                    discovered = True
                    discovery_price_peak = price_t_1
                    
        elif profile_type == 'believer':
            if not discovered:
                if is_benchmark_asset:
                    excess_return = ret_60d_asset_t_1
                else:
                    excess_return = ret_60d_asset_t_1 - ret_60d_bench_t_1
                    
                if excess_return > 0.15:
                    discovered = True
                    
        # 2. 信号转变记录
        if discovered and not prev_discovered:
            transitions.append(1)
        elif not discovered and prev_discovered:
            transitions.append(-1)
        else:
            transitions.append(0)
            
        # 3. 仓位暴露与动态调节配置
        exp_bull = params.get('exposure_bull', max_lev)
        oscillate_w = params.get('oscillate_weight', 0.5)
        
        if profile_type == 'non_believer':
            default_osc = exp_bull * oscillate_w
            default_bear = 0.0
        else:
            default_osc = 0.8 + (exp_bull - 0.8) * oscillate_w if exp_bull > 0.8 else exp_bull
            default_bear = 0.8
            
        exp_osc = params.get('exposure_oscillate', default_osc)
        exp_bear = params.get('exposure_bear', default_bear)
        
        if discovered:
            if profile_type == 'non_believer':
                if feel_state == 'bull':
                    exp = exp_bull
                elif feel_state == 'oscillate':
                    exp = exp_osc
                else: # bear
                    exp = exp_bear
                    if exp == 0.0:
                        discovered = False
            elif profile_type == 'believer':
                if feel_state == 'bull':
                    exp = exp_bull
                elif feel_state == 'oscillate':
                    exp = exp_osc
                else: # bear
                    exp = exp_bear
            states.append('discovered')
        else:
            exp = 0.0
            states.append('cash')
            
        exposures.append(exp)
        
    exposures = pd.Series(exposures, index=target_index)
    transitions = pd.Series(transitions, index=target_index)
    
    # 模拟组合NAV
    portfolio_returns = []
    
    if leverage_type == 'daily_etf':
        etf_ret = LeverageSimulator.simulate_daily_etf(returns, max_lev, interest_rate, fee_rate_annual)
        for t in range(len(target_index)):
            exp = exposures.iloc[t]
            ret_t = returns.iloc[t]
            if exp <= 1.0:
                r_p = exp * ret_t + (1.0 - exp) * r_cash
            else:
                if max_lev > 1.0:
                    w_etf = (exp - 1.0) / (max_lev - 1.0)
                else:
                    w_etf = 0.0
                w_1x = 1.0 - w_etf
                r_p = w_1x * ret_t + w_etf * etf_ret.iloc[t]
            portfolio_returns.append(r_p)
            
    elif leverage_type in ['margin_constant', 'margin_static']:
        if leverage_type == 'margin_constant':
            for t in range(len(target_index)):
                exp = exposures.iloc[t]
                ret_t = returns.iloc[t]
                if exp <= 1.0:
                    r_p = exp * ret_t + (1.0 - exp) * r_cash
                else:
                    interest_daily = interest_rate / 365.0
                    friction = exp * abs(exp - 1.0) * abs(ret_t) * 0.001
                    r_p = exp * ret_t - (exp - 1.0) * interest_daily - friction
                portfolio_returns.append(r_p)
        else: # margin_static
            account = MarginAccount(initial_equity=100.0, interest_rate_annual=interest_rate, trans_fee_rate=0.001)
            prev_exp = 1.0
            nav_series = [100.0]
            
            for t in range(1, len(target_index)):
                exp = exposures.iloc[t]
                ret_t = returns.iloc[t]
                price_t_1 = prices.iloc[t-1]
                
                if exp != prev_exp:
                    account.rebalance(exp, price_t_1)
                    prev_exp = exp
                    
                state = account.update_daily(ret_t)
                nav_series.append(state['equity'])
                
            nav_s = pd.Series(nav_series, index=target_index)
            portfolio_returns = nav_s.pct_change().fillna(0.0)
            
    portfolio_returns = pd.Series(portfolio_returns, index=target_index)
    nav_s = (1.0 + portfolio_returns).cumprod()
    
    result_df = pd.DataFrame({
        'nav': nav_s,
        'exposure': exposures,
        'state': states,
        'transition': transitions,
        'close': prices
    }, index=target_index)
    
    return result_df

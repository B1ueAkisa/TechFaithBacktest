import pandas as pd
import numpy as np

class LeverageSimulator:
    """
    杠杆模拟器类，支持模拟杠杆ETF和融资账户。
    """
    
    @staticmethod
    def simulate_daily_etf(returns: pd.Series, leverage: float, interest_rate_annual: float, fee_rate_annual: float = 0.0095) -> pd.Series:
        """
        模拟每日重置型杠杆 ETF (包含内嵌的衍生品掉期融资利息成本和管理费)
        R_etf,t = L * R_t - (L - 1) * r_borrow_daily - fee_daily
        """
        interest_daily = interest_rate_annual / 365.0
        fee_daily = fee_rate_annual / 252.0
        
        # 实际杠杆 ETF 并不直接持股，而是通过场外互换(OTC Swap)向投行借入资金买入多倍资产。
        # 借资部分 (L - 1) 需向投行支付利息成本（通常略高于美联储基准利率，如 5.5% 左右），这是实际 TQQQ 损耗极大的主因。
        borrow_cost = (leverage - 1.0) * interest_daily if leverage > 1.0 else 0.0
        
        etf_returns = leverage * returns - borrow_cost - fee_daily
        etf_returns = etf_returns.clip(lower=-0.999)
        return etf_returns

    @staticmethod
    def simulate_constant_margin(returns: pd.Series, leverage: float, interest_rate_annual: float, fee_rate_trans: float = 0.001) -> pd.Series:
        """
        模拟恒定融资杠杆（每日重置调仓维持恒定杠杆比率）
        """
        interest_daily = interest_rate_annual / 365.0
        friction = leverage * abs(leverage - 1.0) * returns.abs() * fee_rate_trans
        
        margin_returns = leverage * returns - (leverage - 1.0) * interest_daily - friction
        margin_returns = margin_returns.clip(lower=-0.999)
        return margin_returns


class MarginAccount:
    """
    静态融资账户模拟器，支持不定期调仓（按策略信号调仓）
    """
    def __init__(self, initial_equity: float = 100.0, interest_rate_annual: float = 0.065, 
                 maintenance_ratio: float = 1.30, trans_fee_rate: float = 0.001):
        self.initial_equity = initial_equity
        self.interest_rate_annual = interest_rate_annual
        self.maintenance_ratio = maintenance_ratio
        self.trans_fee_rate = trans_fee_rate
        
        self.reset()
        
    def reset(self):
        self.equity = self.initial_equity
        self.debt = 0.0
        self.assets = self.initial_equity
        self.leverage = 1.0
        self.liquidated = False
        
    def rebalance(self, target_leverage: float, asset_price: float):
        if self.liquidated:
            return
            
        target_assets = self.equity * target_leverage
        trade_volume = abs(target_assets - self.assets)
        fee = trade_volume * self.trans_fee_rate
        
        self.equity -= fee
        if self.equity <= 0:
            self.equity = 0.0
            self.assets = 0.0
            self.debt = 0.0
            self.leverage = 0.0
            self.liquidated = True
            return
            
        self.assets = self.equity * target_leverage
        self.debt = self.assets - self.equity
        self.leverage = target_leverage

    def update_daily(self, asset_return: float) -> dict:
        if self.liquidated:
            return {
                "equity": 0.0,
                "assets": 0.0,
                "debt": 0.0,
                "leverage": 0.0,
                "collateral_ratio": 0.0,
                "liquidated": True
            }
            
        self.assets = self.assets * (1.0 + asset_return)
        
        interest_daily = (self.interest_rate_annual / 365.0) * self.debt
        self.debt += interest_daily
        
        self.equity = self.assets - self.debt
        
        if self.equity > 0:
            self.leverage = self.assets / self.equity
        else:
            self.leverage = np.inf
            
        if self.debt > 0:
            collateral_ratio = self.assets / self.debt
        else:
            collateral_ratio = np.inf
            
        if self.debt > 0 and collateral_ratio < self.maintenance_ratio:
            self.liquidated = True
            liquidation_fee = self.assets * self.trans_fee_rate
            self.equity = max(0.0, self.assets - self.debt - liquidation_fee)
            self.assets = 0.0
            self.debt = 0.0
            self.leverage = 0.0
            collateral_ratio = 0.0
            
        return {
            "equity": self.equity,
            "assets": self.assets,
            "debt": self.debt,
            "leverage": self.leverage,
            "collateral_ratio": collateral_ratio,
            "liquidated": self.liquidated
        }


def calculate_metrics(nav_series: pd.Series) -> dict:
    returns = nav_series.pct_change().dropna()
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1.0
    
    n_days = len(nav_series)
    annual_return = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (252.0 / n_days) - 1.0
    
    annual_vol = returns.std() * np.sqrt(252)
    
    rf = 0.02
    sharpe = (annual_return - rf) / annual_vol if annual_vol > 0 else 0
    
    cum_max = nav_series.cummax()
    drawdown = (nav_series - cum_max) / cum_max
    max_dd = drawdown.min()
    
    calmar = annual_return / abs(max_dd) if max_dd != 0 else np.nan
    
    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar
    }
